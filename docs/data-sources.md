# 真实数据来源与接入说明

本文列出 Flight Forecast Lab 面向真实训练所设计的官方数据来源。仓库的合成演示不自动下载这些数据，也不随代码分发外部数据文件。

## 来源总览

| 任务 | 官方来源 | 主要用途 | 关键限制 |
| --- | --- | --- | --- |
| 票价 | [BTS Origin and Destination Survey（DB1B / DB1C）](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data) | 历史已售客票、航线、承运人、距离 | 抽样调查，不是实时搜索报价 |
| 准点 | [BTS Reporting Carrier On-Time Performance](https://transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=&gnoyr_VQ=FGJ) | 航班级计划/实际到达、取消、延误 | 报告范围有限，发布有滞后 |
| 天气 | [NOAA/NCEI GHCN Hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) | 历史温度、降水、风等天气观测 | 观测值不能直接当作未来可知特征 |

访问前请查看各官方网站的最新字段说明、发布日期、服务限制与使用条款。下载文件应保留原始文件名、下载时间、来源 URL 和校验哈希。

## 1. BTS O&D：DB1B 与 DB1C

### 制度与粒度

BTS 官方说明中，DB1B 是报告承运人客票的 10% 样本，历史上按季度发布；该收集制度在 2025 年 7 月停止。DB1C 自 2025 年 7 月起采用 40% 样本并按月收集。跨越这一时间点的训练集必须记录来源制度，避免把采样率或月/季度粒度变化误当作市场变化。

官方入口：

- [DB1B 数据库概览与表说明](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFI&Yv0x=D)
- [DB1C O&D 总入口](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data)
- [DB1C Ticket 文件](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-ticket)
- [DB1C Market 文件](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-market)
- [DB1C Coupon 文件](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-coupon)
- [DB1C Segment 文件](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-segment)

### 票价建模建议

优先选取能够明确解释为旅客支付金额的 Ticket/Market 字段，并保留乘客权重、往返标志、行程结构和数据质量标志。不同表中的“fare”含义可能不同，不要只按相似列名合并。

最低清洗要求：

1. 使用官方键连接 Ticket、Market、Coupon/Segment；先验证连接基数，防止一对多连接把样本和乘客权重放大。
2. 过滤或单独标记不合理票价、零/负距离、未知机场、非目标币种语义和缺失承运人。
3. 明确票价是单程、往返还是按市场分摊；不要把不同语义放在同一目标列。
4. 保留 `passengers` 作为样本权重候选，但不得先按它复制海量行。
5. DB1B/DB1C 通常不能提供实时搜索场景中的准确询价提前期。若没有合法的报价快照数据，不应声称模型学到了“今天买还是下周买”的因果规律。

### 向统一契约映射

O&D 数据可直接或经派生提供 `origin`、`destination`、`airline`、`stops`、`distance_km` 和统一目标列 `price_usd`。`cabin`、精确 `departure_time`、`quote_time` 与 `duration_minutes` 的可用性需按具体表和目标样本验证；缺失字段不能用目标信息反推。

若公共数据只有月或季度粒度，应保留 `source_period_start` / `source_period_end`，并使用能表达这种粒度的训练流程。不要为每条记录捏造一个精确起飞时间。

## 2. BTS On-Time Performance

BTS 的 [准点表现说明页](https://www.transtats.bts.gov/ot_delay/) 使用“到达比计划晚 15 分钟或以上”为延误标准。本项目与之保持一致，并额外要求航班未取消：

```text
on_time = (Cancelled != 1) AND (ArrDelay < 15)
```

建议下载字段：

- 身份与时间：`FlightDate`、报告/运营承运人、航班号；
- 航线：`Origin`、`Dest`、距离；
- 计划时间：计划出发、计划到达；
- 结果：`ArrDelay`、`Cancelled`、`Diverted`；
- 可选诊断：延误原因、滑行时间、尾号等，但预测时刻未知的结果字段不得进入特征。

数据处理注意事项：

1. 将 BTS 本地计划时间结合机场时区转换为带时区时间；不要把 `HHMM` 直接当普通整数。
2. `2400`、跨午夜到达、夏令时切换和机场时区变更需要专门测试。
3. 取消航班的 `ArrDelay` 往往缺失，但标签仍应为 0。
4. 备降样本的标签策略必须预先声明。推荐在主评估中作为负类或单独报告，不能因结果不便而静默删除。
5. 为避免记忆同一航班的后验信息，按日期切分，并对承运人代码变更进行版本化处理。

## 3. NOAA / NCEI 天气

历史小时天气建议使用 [NCEI GHCN Hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly) 的 HTTPS 批量文件；未来短期航空天气可从 [NOAA Aviation Weather Center Data API](https://aviationweather.gov/data/api/) 获取 METAR/TAF。GHCN Hourly 批量下载与 Aviation Weather API 当前均不要求 API key，但调用方仍须遵守官方频率限制、标识要求和最新条款。

建议的连接流程：

1. 建立版本化的机场到气象站映射，保留站点距离、海拔和有效日期；
2. 将航班计划时间统一转换为 UTC，再匹配该时刻之前最近的可用天气；
3. 对温度、降水、能见度、风速、阵风、雷暴或降雪等字段统一单位；
4. 记录观测发布时间或可用时间，而不只是观测发生时间；
5. 缺测时保留缺失指示，不用未来观测向前填充。

### 防止天气泄漏

NOAA 历史观测适合建立“天气与延误关系”的回溯基线，但预测未来航班时，实际天气尚不可知。生产系统应接入在预测时刻真实可获得的预报快照，并使用同一预报时效训练。若项目只接入历史观测，API 中的 `weather_severity_forecast` 必须由调用方提供，且文档和 UI 应明确它是外部预测输入。

## 4. 建议的数据落地层

不要直接在下载压缩包上训练。建议保存三层：

```text
data/raw/<source>/<release>/        原始文件，只读保留
data/interim/<source>/<version>/    解压、字段标准化、质量报告
data/processed/<dataset_version>/   符合项目数据契约的训练表
```

每个数据版本至少记录：

- 来源 URL 与下载时间；
- 官方发布日期或覆盖周期；
- 原始文件 SHA-256；
- 适配器代码版本；
- 输入/输出行数和连接基数；
- 删除、修正和缺失值规则；
- 标签正类比例；
- 训练、校准、测试的时间边界。

大型原始数据和含许可限制的数据不应直接提交到 Git。可提交小型、去标识、来源明确的测试夹具，以及生成它们的说明。

## 5. 合并边界

票价和准点数据的观测单位不同：O&D 记录描述客票/市场/行程，而 On-Time 记录描述实际航班。不要仅凭起点、终点、航司和月份做多对多连接后把它称为同一旅程。

本项目将两个任务建成独立模型是有意设计：它们可以共享经过验证的静态特征，但不要求每张客票与某一实际航班强行对应。若拥有航班号、日期和完整行程键，应先验证唯一性和覆盖率，再考虑更细连接。
