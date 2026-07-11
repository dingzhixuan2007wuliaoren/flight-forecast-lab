# 数据与 API 契约

本文定义 Flight Forecast Lab 的稳定边界。原始 BTS/NOAA 字段应先通过适配器转换到这里的语义，再进入训练或推理。运行中的 FastAPI `/docs` 是具体 JSON 模式的最终依据。

## 通用约定

- 字段名使用 `snake_case`；CSV 使用 UTF-8，推荐处理后数据使用 Parquet。
- 日期使用 ISO 8601 `YYYY-MM-DD`；时间戳必须含 UTC 偏移，例如 `2026-08-15T13:30:00-04:00` 或 `2026-08-15T17:30:00Z`。
- 机场使用大写三位字母或数字代码。通常是 IATA 风格代码，但训练数据必须保留代码体系与有效期元数据。
- 航司代码统一大写；代码复用或历史更名应由带有效日期的映射表处理。
- 距离统一为公里，时长统一为分钟，票价统一为美元。
- 比率/指数限定在 `[0, 1]`，除非字段另有说明。
- 缺失值使用真正的 null/NA，不使用空字符串、`-999`、`UNKNOWN` 混充数值。
- 推理请求不得包含训练标签或结果发生后才可知的字段。

## 票价预测 API

端点：`POST /v1/predict/price`

### 请求字段

| 字段 | 类型 | 必填 | 约束与语义 |
| --- | --- | --- | --- |
| `origin` | string | 是 | 起飞机场，3 位字母或数字；服务会转为大写 |
| `destination` | string | 是 | 到达机场，3 位字母或数字；不得与起点相同 |
| `airline` | string | 是 | 2–3 位字母或数字；营销/票务口径在训练与推理间必须一致 |
| `cabin` | string | 否 | `economy`、`premium_economy`、`business` 或 `first`；默认 `economy` |
| `stops` | integer | 否 | 经停/转机次数，范围 0–3；默认 0 |
| `duration_minutes` | integer | 是 | 计划总行程时长，`30 < value <= 1800` |
| `distance_km` | number | 是 | 行程或市场距离，`50 < value <= 20000` |
| `quote_time` | datetime | 是 | 本次价格预测/询价时刻，必须带时区 |
| `departure_time` | datetime | 是 | 计划起飞时刻，必须带时区、晚于 `quote_time` 且不超过其后 370 天 |

训练表在上述字段基础上增加：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `price_usd` | number | 回归目标；与行程范围一致的旅客支付/报价金额，必须为正 |
| `sample_weight` | number / null | 可选，例如 O&D 的旅客数；必须为正且不得导致重复加权 |
| `source` | string | 数据来源，如 `synthetic`、`bts_db1b`、`bts_db1c` |
| `source_record_id` | string / null | 可追溯但不进入模型的源记录键 |
| `source_period_start` | date / null | 来源只有月/季度粒度时的周期开始 |
| `source_period_end` | date / null | 来源只有月/季度粒度时的周期结束 |

### 输出语义

响应字段为：

- `estimated_price_usd`：美元价格点估计；
- `interval_80_low_usd` / `interval_80_high_usd`：80% 区间下界和上界；
- `days_until_departure`：由两个输入时间戳计算的提前天数；
- `model_version`：产物中的模型版本；
- `warning`：非实时报价与非最低价保证提示。

服务保证 `interval_80_low_usd <= estimated_price_usd <= interval_80_high_usd`，且下界不小于 0。

区间针对单次未来价格观测，不应解释为“80% 概率价格一定落在此范围”的无条件保证；其覆盖依赖部署分布与校准分布的一致性。

## 准点预测 API

端点：`POST /v1/predict/on-time`

### 请求字段

| 字段 | 类型 | 必填 | 约束与语义 |
| --- | --- | --- | --- |
| `origin` | string | 是 | 起飞机场，3 位字母或数字；服务会转为大写 |
| `destination` | string | 是 | 到达机场，3 位字母或数字；不得与起点相同 |
| `airline` | string | 是 | 2–3 位字母或数字；报告/运营口径在训练与推理间必须一致 |
| `distance_km` | number | 是 | 计划航段距离，`50 < value <= 20000` |
| `scheduled_departure` | datetime | 是 | 计划起飞时刻，必须带时区 |
| `weather_severity_forecast` | number | 否 | 预测时刻可获得的天气严重度，范围 `[0, 1]`；默认 0.2，0 最轻、1 最重 |
| `origin_congestion_index` | number | 否 | 预测时刻可获得的起飞机场拥堵指数，范围 `[0, 1]`；默认 0.4 |

训练 CSV 中，以上七个特征列全部必填（API 默认值不会自动补入 CSV），并增加原始结果与标签：

| 字段 | 类型 | 必填 | 语义 |
| --- | --- | --- | --- |
| `cancelled` | boolean | 是 | 是否取消 |
| `arrival_delay_minutes` | number / null | 条件必填 | 未取消且有最终到达记录时的实际到达延误；提前到达为负数 |
| `on_time` | integer | 是 | 二分类目标，只能为 0 或 1 |
| `source` | string | 是 | 如 `synthetic`、`bts_on_time` |
| `source_record_id` | string / null | 否 | 仅用于追溯，不进入模型 |

标签必须按以下唯一规则生成：

```text
if cancelled:
    on_time = 0
else if arrival_delay_minutes is present:
    on_time = 1 when arrival_delay_minutes < 15, otherwise 0
else:
    on_time = missing  # 进入隔离/审查，不得默认填 1
```

备降航班需要数据版本级策略。若没有可靠的最终到达延误，推荐从主训练集隔离并单独报告数量；任何不同策略都必须写入元数据。

### 输出语义

响应字段为 `on_time_probability`、`disruption_probability`、`risk_level`、`definition` 和 `model_version`。两个概率均在 `[0, 1]` 且互为补数；当前风险分档为准点概率 `>= 0.80` 时 `low`，`>= 0.60` 且 `< 0.80` 时 `medium`，否则 `high`。

风险分档阈值不是准点定义中的 15 分钟阈值：前者把模型概率转成面向用户的风险级别，后者从真实到达结果生成训练标签。

## 服务辅助端点

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 进程与模型加载健康状态；不代表模型仍然准确 |
| `GET /v1/model-info` | 模型版本、训练来源、时间和可用任务信息 |

健康响应不得暴露本地绝对路径、密钥或原始训练记录。模型信息应能让调用方识别是否误用了 `synthetic` 演示模型。

## 派生特征规则

推荐从原始请求一致地派生以下信息，而不是要求调用方重复提供：

- `lead_time_hours = departure_time - quote_time`；
- 起飞月份、星期、小时和周末标志；
- 标准化航线键（方向性是否保留需由任务声明）；
- 距离/时长的一致性诊断。

所有时间派生应使用字段自身的时区语义。不能先丢弃偏移再计算提前期；夏令时边界需要自动化测试。

## 数据质量拒收规则

以下记录不得静默进入训练：

- 起点等于终点、机场/航司代码为空或无法解析；
- 非有限数值、非正距离/时长/票价；
- `departure_time <= quote_time`；
- 指数超出 `[0, 1]`；
- 取消但 `on_time == 1`；
- 未取消、到达延误不少于 15 分钟但 `on_time == 1`；
- 同一 `source_record_id` 出现冲突目标；
- 源表连接后行数异常膨胀；
- 使用实际到达、实际天气或延误原因作为未来预测输入。

适配器应输出质量报告：输入行数、接收行数、各拒绝原因数量、缺失率、重复率、标签比例和时间范围。

## 版本与兼容性

每个训练数据集和模型应记录一个 `schema_version`。建议语义：

- 补充可选字段：向后兼容的小版本；
- 改变单位、枚举、标签或必填字段：不兼容的大版本；
- 适配器必须拒绝未知的大版本，而不是猜测转换。

API 客户端应以 `/v1/` 为稳定主版本边界，并读取 `/v1/model-info` 判断模型来源与版本。批量推理时还应保存请求模式版本、模型版本和预测时间，以便复现。
