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

NOAA 历史观测适合建立“天气与延误关系”的回溯基线，但预测未来航班时，实际天气尚不可知。生产系统应保存预测时刻真实可获得的预报快照，并使用相同预报时效训练。本项目的服务端会自动解析 Open-Meteo/NOAA 天气信号；演示版数据源不可用时返回明确标记的合成模型先验，不冒充真实历史均值，也不要求用户填写天气严重度。

## 4. 运行时免费上下文来源

比较和准点接口还会在运行时查询以下免费来源；这些信号是短期上下文，不会被伪装成实时可售票价：

| 信号 | 优先来源 | 无法获取时 |
| --- | --- | --- |
| 全球机场坐标 | [OurAirports public-domain data](https://ourairports.com/data/) | 内置主要全球机场目录；未知代码返回校验错误 |
| 当前天气与预报 | [Open-Meteo](https://open-meteo.com/en/docs) 当前模型天气/小时预报与 [NOAA Aviation Weather](https://aviationweather.gov/data/api/) METAR/TAF | 同月训练平均值或季节模型先验，并明确标为 `proxy` |
| 机场运行 | 美国机场使用无需密钥的 [FAA NAS Status](https://nasstatus.faa.gov/) 当前事件；其他机场配置免费 AirLabs key 后使用 [AirLabs schedules](https://airlabs.co/docs/schedules) | [ADSB.lol](https://www.adsb.lol/docs/open-data/api/) 当前飞机密度代理；目标时刻不适用时回退训练平均值/合成先验 |
| 时事新闻 | 无需密钥的 [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) 近七日中断类新闻（`DateDesc`）；DOC 失败时使用官方 [GAL 滚动 RSS](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/) | 先返回不超过 6 小时、降低影响且标为 `historical` 的航线缓存；无缓存时返回中性值且不虚构文章 |
| 航班/航线详情 | 配置免费 key 后，优先使用 [AirLabs schedules](https://airlabs.co/docs/schedules) 近实时航班计划，再使用 [AirLabs routes](https://airlabs.co/docs/routes) 周期时刻表投影 | 模型航段距离/时长；中转另加 90 分钟假设，但不生成航班号或钟点 |

GDELT DOC 属于近实时来源，全球数据通常约每 15 分钟更新；GAL RSS 每分钟更新并滚动保留最近约 15 分钟的链接。服务按航线缓存成功结果 15 分钟以减少重复请求。文章字段中的时间表示 GDELT 观察 / 索引时间，不保证是媒体的准确发布时间；标题保持来源语言。使用或再分发 GDELT 数据时须注明并链接 [GDELT Project](https://www.gdeltproject.org/about.html)。

机场运行必须区分“当前快照”和“适用于目标起飞时刻的模型信号”。FAA NAS Status 是美国当前运行事件的权威来源，但机场没有出现在事件列表中只表示没有列出的当前 FAA 事件；不能推导出所有航班正常。范围有限的 `freeForm` 限制不能当作完整机场关闭。FAA 事件只有在时间范围覆盖计划起飞时才可进入目标信号。

AirLabs 免费 `schedules` 是近实时接口，结果按当前或目标时刻前后 90 分钟筛选，并受最多 50 行、免费配额和提供方可见计划时段限制，因此是实际航班样本而不是完整机场统计。航班比较的 schedules/routes 查询也各自最多 50 行；任一响应的 `request.has_more` 为真或达到 50 行时，`schedule_sample_truncated=true`，表示真实航班列表可能不完整。本演示为控制免费配额和调用量不继续分页。`routes` 表达按星期重复的周期时刻表；系统可以把完整记录投影到所选日期，但必须标为 `recurring_timetable_projection`，不能称为当天已确认执行的实时航班。ADSB.lol 只统计机场坐标附近的飞机并生成密度代理；它不提供真实延误、取消、机场容量或官方地面管制数据，且只有计划起飞距查询时刻不超过 90 分钟时才允许作为目标信号。其他情况继续使用明确标记的训练平均值/合成先验。

比较请求只提交 `departure_date`，且日期不得早于出发机场当地今天、并在当地今天后 370 天以内。未来日期使用当地正午作为模型、天气和新闻参考；同日若正午仍有超过 30 分钟余量则使用正午，否则沿 UTC 时间线推进 30 分钟再转换回机场当地时间。安全参考跨入次日则返回 422。`departure_time_basis` 明确区分正午与剩余同日参考，两者都不是航班钟点。只有通过路线、日期、时区、时长、未来起飞及状态校验的 AirLabs live 行才可显示航班号及起降时间；取消或已起飞的 live identity 会阻止相同 routes 投影重新出现。Live schedules 的航站楼仍是 provider-estimated terminal，不是当天确认值；Routes 的 possible terminals 和 last-used aircraft 不返回。无适用数据时，仅当该航司存在不同于起终点的映射枢纽才返回一站模型；否则使用 `model_route_unresolved`、`stops=null` 和 `legs=[]`，总距离/时长只是 O&D 模型参考。系统不补造航段、航班号、钟点，也不使用其他航司或通用枢纽。舱位始终保持 `catalog_scenario`。

主页每个 offer 都链接到 `GET /details/offer`，并通过 `POST /v1/offer-detail` 获取 schedule 或模型行程依据。天气和新闻详情页继续使用各自的上下文接口。

天气与新闻详情页从主页接收 `departure_date`，每次刷新由服务重新计算带 `departure_time_basis` 的模型/上下文参考，避免沿用已过期的同日固定时刻；这些时间不是航班计划。天气详情接口在出发机场参考时刻和到达机场模型估算抵达参考分别查询 Open-Meteo，并显示当前条件、目标小时、前后 12 小时趋势和风险拆解；可用时还显示 NOAA METAR/TAF 原始报文与自动解释。新闻详情接口最多显示最近 7 天的 20 篇匹配文章及分类、命中词和时效权重。网页分别每 10 分钟和 15 分钟刷新，但服务端同样使用短期缓存以遵守免费来源的负载与配额边界。

详细的状态、缓存、严格政策字段和失败回退语义见 [`runtime-context.md`](runtime-context.md)。免费服务的配额、覆盖和条款可能变化，公开部署前应重新核对官方说明。

### 配置可选的免费 AirLabs key

在启动服务的同一个 PowerShell 窗口设置：

```powershell
$env:AIRLABS_API_KEY="your-free-key"
python -m flight_forecaster serve --model-dir artifacts/demo
```

应用读取进程环境变量，不会自动加载 `.env`。不要把 key 写入源码、浏览器参数、前端 JavaScript 或提交到 GitHub。未配置 key 是受支持的运行方式；此时 API 使用明确标记的 `model_fallback`，不会伪造 provider 数据。

## 5. 建议的数据落地层

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

## 6. 合并边界

票价和准点数据的观测单位不同：O&D 记录描述客票/市场/行程，而 On-Time 记录描述实际航班。不要仅凭起点、终点、航司和月份做多对多连接后把它称为同一旅程。

本项目将两个任务建成独立模型是有意设计：它们可以共享经过验证的静态特征，但不要求每张客票与某一实际航班强行对应。若拥有航班号、日期和完整行程键，应先验证唯一性和覆盖率，再考虑更细连接。
