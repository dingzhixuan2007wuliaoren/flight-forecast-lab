# 数据与 API 契约

本文定义 Flight Forecast Lab 的稳定边界。原始 BTS/NOAA 字段应先通过适配器转换到这里的语义，再进入训练或推理。运行中的 FastAPI `/docs` 是具体 JSON 模式的最终依据。

## 通用约定

- 字段名使用 `snake_case`；CSV 使用 UTF-8，推荐处理后数据使用 Parquet。
- 日期使用 ISO 8601 `YYYY-MM-DD`；单项接口时间戳必须含 UTC 偏移，例如 `2026-08-15T13:30:00-04:00` 或 `2026-08-15T17:30:00Z`。比较接口另允许无偏移墙钟时间，并按出发机场当地时区解释。
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
| `departure_time` | datetime | 是 | 计划起飞时刻，必须带时区、晚于服务接收时刻且不超过其后 370 天；`quote_time` 已从公开请求删除 |

服务根据 `origin`、`destination` 和 `stops` 自动生成 `distance_km` 与 `duration_minutes`，常规请求不得手工覆盖。训练表仍须包含真实或经过验证的距离与时长，并在上述字段基础上增加：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `price_usd` | number | 回归目标；与行程范围一致的旅客支付/报价金额，必须为正 |
| `duration_minutes` | integer | 训练特征；计划总行程时长，`30 < value <= 1800` |
| `distance_km` | number | 训练特征；行程或市场距离，`50 < value <= 20000` |
| `news_disruption_index` | number | 报价时刻可获得的近期航线新闻风险快照，范围 `[0, 1]`；不得使用事后新闻 |
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
- `distance_km` / `duration_minutes`：服务自动推算并实际送入模型的航程距离与时长；
- `model_version`：产物中的模型版本；
- `warning` / `warning_en`：新闻风险已纳入模型、但结果并非实时报价或最低价保证的中英文提示。

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
| `scheduled_departure` | datetime | 是 | 计划起飞时刻，必须带时区、晚于当前时刻且不超过其后 370 天 |

调用方不再输入距离、天气严重度或机场拥堵。服务会根据起点、终点与计划时间自动解析距离，并获取预测时刻可用的天气、机场运行和近期新闻信号；免费数据源不可用或不适用于计划日期时，演示版使用仅由训练切分计算、明确标为 `proxy` 的同月天气/机场平均值，新闻则中性回退。默认产物来源标为 `synthetic_demo_training_average`；部署方只有在接入并审计真实历史聚合数据后才能标为 `historical`。训练 CSV 仍须保存当时真实可获得的上下文快照，并增加原始结果与标签：

| 字段 | 类型 | 必填 | 语义 |
| --- | --- | --- | --- |
| `cancelled` | boolean | 是 | 是否取消 |
| `distance_km` | number | 是 | 训练特征；计划航段距离，`50 < value <= 20000` |
| `weather_severity_forecast` | number | 是 | 当时可获得的天气预报严重度快照，范围 `[0, 1]` |
| `origin_congestion_index` | number | 是 | 当时可获得的出发机场运行/拥堵快照，范围 `[0, 1]` |
| `news_disruption_index` | number | 是 | 当时可获得的近七日航线中断新闻风险，范围 `[0, 1]` |
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

响应字段为 `on_time_probability`、`disruption_probability`、自动推算的 `distance_km`、`risk_level`、中文 `definition`、英文 `definition_en` 和 `model_version`。两个概率均在 `[0, 1]` 且互为补数；当前风险分档为准点概率 `>= 0.80` 时 `low`，`>= 0.60` 且 `< 0.80` 时 `medium`，否则 `high`。

风险分档阈值不是准点定义中的 15 分钟阈值：前者把模型概率转成面向用户的风险级别，后者从真实到达结果生成训练标签。

## 航司与舱位比较 API

端点：`POST /v1/compare`

### 请求字段

| 字段 | 类型 | 必填 | 约束与语义 |
| --- | --- | --- | --- |
| `origin` | string | 是 | 起飞机场 IATA 风格三位代码 |
| `destination` | string | 是 | 到达机场 IATA 风格三位代码；不得与起点相同 |
| `departure_time` | datetime | 是 | 未来计划时间；无偏移时按出发机场当地时区解释，有偏移时视为绝对时刻；不得超过当前时间后 370 天 |

### 响应结构

- `context.weather`、`context.operations`、`context.news`：自动获取的三个上下文信号，均含 `[0, 1]` 数值、来源、获取时间、双语摘要和明确状态；新闻还可含最多五篇去重后的来源文章。`context.operations` 顶层是实际进入目标起飞时刻模型的信号，不能与嵌套的查询时刻 `current_snapshot` 混为一谈。
- `offers`：逐航司、逐支持舱位的模型估价、80% 区间、行程时长、准点概率、风险等级、行李/学生/改签/退票状态及学生验证说明；`cabin_status=catalog_scenario` 明确表示舱位来自比较目录而非实时报价确认。
- `rankings.direct_first`、`rankings.lowest_price`、`rankings.student_first`：引用 `offers[].id` 的完整排序。
- `departure_time` / `departure_timezone`：服务按出发机场坐标解析后的带偏移时间和 IANA 时区；`warnings`：中英文限制说明；`model_version`：本次比较所用模型版本。

上下文状态只允许 `live`、`forecast`、`proxy`、`historical`、`neutral` 或 `unavailable`。`route_status=provider_confirmed` 表示免费航线提供方确认该航司经营直飞航线；`model_scenario` 是一站中转比较场景，不能解释为真实可售航班。比较始终保留全球目录中的所有航司；提供方返回的其他航司也会追加到结果中。

`context.operations` 与可选的 `context.operations.current_snapshot` 使用以下解释字段：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `value` / `status` / `source` / `observed_at` | 通用信号字段 | `[0,1]` 风险/压力、来源状态、来源名与观测/获取时间 |
| `method` | string | 计算方法，例如 FAA 事件、实际出港样本、飞机密度、训练平均值或合成先验 |
| `data_tier` | string | 证据层级，例如权威当前事件、实际航班样本、当前代理或先验 |
| `applicability` | string | `current_only`、`target_departure` 或 `target_departure_prior` 等时间适用范围 |
| `metrics[]` | array | `key`、数值/文本 `value` 与可选 `unit`；例如事件数、延误样本数或飞机数 |
| `events[]` | array | 事件类型、`[0,1]` 严重度、原因、起止时间和可选影响范围 |
| `window_start` / `window_end` | datetime / null | 事件或样本用于解释的时间窗口 |
| `sample_size` / `sample_limit` / `sample_truncated` | integer / integer / boolean | 样本量、提供方行数上限与是否可能截断 |
| `fallback_reason` | string / null | 当前来源缺失或降级的机器可读原因组合 |
| `current_snapshot` | object / null | 仅顶层目标信号拥有；查询时刻的独立快照，不保证适用于目标起飞时刻 |

美国机场的 FAA `freeForm` 限制只作为低严重度 `restriction` 当前事件，不能当作全机场关闭，也不会进入目标信号。ADSB.lol 的 `traffic_density` 只代表附近飞机密度，不能解释为真实延误率、取消率或官方机场管制状态。AirLabs 免费出港样本以目标/当前时刻前后 90 分钟过滤，最多请求 50 行；`sample_truncated=true` 时尤其不得把它当作完整机场统计。

`context.news.articles[]` 的字段语义如下：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `title` | string | 来源标题原文；系统不承诺翻译 |
| `url` | HTTP(S) URL | 清理跟踪参数后的来源文章链接 |
| `source` | string | GDELT 返回的来源域名或站点名 |
| `language` | string / null | GDELT 提供时返回的可选来源语言代码 |
| `published_at` | datetime | 兼容字段名；在 GDELT 集成中表示文章被观察 / 索引的时间，不保证等于媒体的准确发布时间 |

GDELT DOC 查询无需密钥，使用近七日窗口与 `DateDesc`；DOC 失败时可使用官方 GAL 滚动 RSS。成功新闻按航线新鲜缓存 15 分钟。实时来源失败时，不超过 6 小时的成功缓存可标为 `historical` 并降低模型影响；没有缓存时必须返回 `neutral`、值 0 和空文章列表。`context.news.observed_at` 表示该新闻上下文快照的获取时间，而不是某篇文章的发布时间。

直飞 offer 使用 `punctuality_basis=direct_leg_model`。一站场景使用 `two_leg_independence_scenario`：其时长在直飞基准上增加 90 分钟，其行程准点率按两个同概率、相互独立航段同时准点的保守场景计算，即 `p_itinerary = p_leg²`。这是显式的模型假设，不是已确认转机方案。

政策状态采用严格证据语义。`unknown` 不等于“不包含”；公开学生计划只标为 `program_available`，不能当成当前航线、日期和舱位已经获得实际学生折扣。只有报价级证据才可使用 `confirmed_free`、`confirmed_included` 或 `confirmed_discount`。

学生优先排序依次比较：最低模型价格、免费托运行李、已确认实际学生折扣、免费改签/退票、年龄与身份验证门槛。`program_available` 本身不满足“实际折扣”条件，只在最后一级使用其公开年龄和验证信息。由于价格是第一排序键，后续条件主要用于同价时打破平局。

## 上下文详情 API

两个详情接口都使用与比较接口相同的三字段请求：`origin`、`destination`、`departure_time`。无偏移时间按出发机场当地时区解释；时间必须在未来且不超过 370 天。页面路由 `GET /details/weather` 与 `GET /details/news` 不直接返回数据，而是读取查询参数并调用下列 API。

### `POST /v1/context/weather-detail`

响应包含航线、带时区的计划起飞、根据航线模型时长估算的抵达时刻、生成时间，以及 `origin_weather` 和 `destination_weather`。每个机场对象包含：

- 机场代码/名称、可选 ICAO、IANA 时区和目标时刻；
- `current` 与 `target` 天气：温度、WMO 天气代码及双语描述、风速、阵风、降水、降水概率、能见度、风险值；
- 目标时刻前后 12 小时最多 25 个 `hourly` 点；
- 最多五项 `risk_components`：天气代码、风、阵风、降水、能见度；
- 当前实现最多返回一条最近有效的 `METAR` 和一条适用于目标时刻的 `TAF`；每条含原始报文、签发/有效时间、风险和保守的双语解释（响应模式为未来扩展保留最多四条容量）；
- `aviation_metadata`：明确区分 NOAA 报告可用、无适用报告、缺少 ICAO、部分产品失败或服务不可用，并给出来源、时间和双语原因；
- `metadata`：总体天气风险的状态、实际主导来源、观测时间、有效期及可选双语回退原因。当适用的 NOAA 报告风险更高时，总体风险和这些时间字段都以该报告为准。

到达时刻是模型估计，不是实际航班计划；自动 METAR/TAF 解读不能替代官方航空气象简报。页面每 10 分钟自动刷新，并提供手动刷新。

### `POST /v1/context/news-detail`

响应包含航线、计划起飞、生成时间、最多 20 篇文章、`route_raw_risk`、`departure_attenuation_factor`、`model_effect`、`model_signal`、元数据、双语摘要与 GDELT 索引时间提示。`route_raw_risk` 是详情页较大文章集的解释性分数；`model_effect` 与 `model_signal.value` 是主预测上下文实际使用的新闻输入，`model_signal` 还给出其状态、来源、观测时间和双语摘要。每篇文章包含原始 `title`、清理后的 HTTP(S) `url`、来源域名、可选语言、`indexed_at`、风险 `category`、最多 12 个 `matched_risk_terms`、`raw_score`、`recency_factor` 和 `weighted_score`。

允许的类别为 `airport_closure`、`airspace_conflict`、`labor_strike`、`extreme_weather`、`cancellation_delay`、`security_cyber` 与 `other_disruption`。文章时间是 GDELT 观察/索引时间，不保证是媒体发布时间；标题保留来源语言。页面每 15 分钟自动刷新，并提供手动刷新。

## 服务辅助端点

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 进程与模型加载健康状态；不代表模型仍然准确 |
| `GET /v1/model-info` | 模型版本、训练来源、时间和可用任务信息 |
| `POST /v1/compare` | 用三个输入生成多航司、多舱位结果与三类完整排序 |
| `GET /details/weather` | 中英双语天气详情页面；查询参数由主页生成 |
| `GET /details/news` | 中英双语新闻详情页面；查询参数由主页生成 |
| `POST /v1/context/weather-detail` | 出发与到达机场的当前/目标天气、趋势和 METAR/TAF 详情 |
| `POST /v1/context/news-detail` | 最近七日最多 20 篇中断新闻及文章级风险解释 |

健康响应不得暴露本地绝对路径、密钥或原始训练记录。模型信息应能让调用方识别是否误用了 `synthetic` 演示模型。

## 派生特征规则

推荐从原始请求一致地派生以下信息，而不是要求调用方重复提供：

- `lead_time_hours = departure_time - quote_time`；
- 起飞月份、星期、小时和周末标志；
- 由出发机场 IANA 时区得到的本地月份、星期和小时；训练表可显式提供 `departure_local_month`、`departure_local_weekday`、`departure_local_hour`，服务推理时会自动生成；
- 标准化航线键（方向性是否保留需由任务声明）；
- 根据机场坐标或版本化航线表推算距离/时长，并保留来源标记；
- 在预测时刻获取天气预报、机场运行和近七日新闻；新闻优先使用 15 分钟航线缓存，失败时只允许使用不超过 6 小时且标为 `historical` 的旧缓存，否则中性回退；
- `news_disruption_index` 同时进入票价与准点模型；天气和机场运行信号进入准点模型；
- 训练数据中的距离/时长一致性诊断。

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

当前演示模型使用 `schema_version = 2`；该版本增加 `news_disruption_index`，并将天气与机场运行输入从客户端请求迁移为服务端自动解析。

API 客户端应以 `/v1/` 为稳定主版本边界，并读取 `/v1/model-info` 判断模型来源与版本。批量推理时还应保存请求模式版本、模型版本和预测时间，以便复现。
