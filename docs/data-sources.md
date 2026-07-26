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

NOAA 历史观测适合建立“天气与延误关系”的回溯基线，但预测未来航班时，实际天气尚不可知。生产系统应保存预测时刻真实可获得的预报快照，并使用相同预报时效训练。本项目的服务端会自动解析 Open-Meteo/NOAA 天气信号；演示版数据源不可用时可展示明确标记的训练平均/合成模型先验作参考，不冒充实时天气，也不要求用户填写天气严重度。票价模型不使用天气；准点率只在状态为 `live` 或 `forecast` 时使用含天气模型，其他状态以 `weather_feature_status=ignored` 切换到无天气模型。

## 4. 运行时免费上下文来源

比较和准点接口还会在运行时查询以下免费来源；这些信号是短期上下文，不会被伪装成实时可售票价：

| 信号 | 优先来源 | 无法获取时 |
| --- | --- | --- |
| 全球机场坐标 | [OurAirports public-domain data](https://ourairports.com/data/) | 内置主要全球机场目录；未知代码返回校验错误 |
| 当前天气与预报 | [Open-Meteo](https://open-meteo.com/en/docs) 当前模型天气/小时预报与 [NOAA Aviation Weather](https://aviationweather.gov/data/api/) METAR/TAF | 同月训练平均值或季节模型先验明确标为 `proxy`，仅作展示参考；准点预测改用无天气模型 |
| 机场运行 | 美国机场使用无需密钥的 [FAA NAS Status](https://nasstatus.faa.gov/) 当前事件；其他机场配置免费 AirLabs key 后使用 [AirLabs schedules](https://airlabs.co/docs/schedules) | OpenSky 匿名当前航迹密度参考，失败或额度耗尽时再用 [ADSB.lol](https://www.adsb.lol/docs/open-data/api/) 飞机密度代理；目标时刻不适用时回退训练平均值/合成先验 |
| 时事新闻 | 无需密钥的 [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) 近七日中断类新闻（`DateDesc`）；DOC 失败时使用官方 [GAL 滚动 RSS](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/) | 先返回不超过 6 小时、降低影响且标为 `historical` 的航线缓存；无缓存时返回中性值且不虚构文章 |
| 严格航班比较 | `auto` 查询所有已配置且获准参与的严格源：[SerpApi Google Flights API](https://serpapi.com/google-flights-api)、SearchAPI.io，以及仅在双开关显式发布后的 `ignav_verified_fares`；每个来源都必须独立验证购票选项，并在账户实际可用额度内尝试其返回的全部合格候选 | 两个或更多来源实际运行时返回 `strict_fare_aggregate` 和逐源 `provider_runs`；相同完整航段与舱位跨源保留最低已验证价格。只有真实额度不足而未验证的部分标为 `partial` / `quota_limited`；全部来源都没有二次验证通过的真实报价时才返回结构化空 `offers`。隔离与仅参考来源绝不补位，结果也不保证全球全量航班 |
| 酒店房型与跨平台评价 | 用户显式操作时调用 [SerpApi Google Hotels Property Details](https://serpapi.com/google-hotels-property-details)；房型来自 `featured_prices[].rooms[]`，平台评分/评价来自 Google 属性字段与 `other_reviews[]` | 仅在名称、紧邻坐标与 provider 房源身份均一致时显示；模糊/多重匹配、额度不足或字段缺失保持不可用，不用其他酒店或估算数据补位 |
| 公共交通 | 无需密钥的 [Transitous MOTIS API](https://transitous.org/api/) 及其[开放 GTFS 来源](https://transitous.org/sources/) | 只显示返回的完整时刻表行程；当地无 feed、无行程或服务失败时返回明确覆盖状态，不推算线路或时长 |

免费来源的额度单位不可混用：SerpApi 按 provider 结算周期执行本地 250 次请求硬墙；SearchAPI.io 是安装/账户生命周期一次性 100 次请求，不是月度额度；Ignav 是一次性最多 1,000 次，默认身份为 `ignav_quarantine`，只有 `IGNAV_STRICT_RELEASE=1` 与 `IGNAV_FREE_ACCOUNT_ATTESTED=1` 同时成立并完成受控验证后才以独立的 `ignav_verified_fares` 身份参与；Scrape.do 每月 1,000 点数、每次参考查询预留 10 点数，只返回聚合覆盖快照；OpenSky 无凭据时默认启用匿名参考，每日 400 API 点数；AeroDataBox 只信任 RapidAPI 免费计划的限额/剩余/重置头来划分账期，没有可信重置证据时执行安装生命周期 600 单位硬墙；AirLabs 所有调用点与进程共用 SQLite 月度账本，配置 key 但没有明确有效的本地上限时 fail closed，响应中的账户月上限/已用量只能收紧硬墙。Scrape.do、OpenSky、AeroDataBox 与 AirLabs 永久只作各自的覆盖、航迹密度、时刻或运行参考，不能单独证明未来机票可售。

GDELT DOC 属于近实时来源，全球数据通常约每 15 分钟更新；GAL RSS 每分钟更新并滚动保留最近约 15 分钟的链接。服务按航线缓存成功结果 15 分钟以减少重复请求。文章字段中的时间表示 GDELT 观察 / 索引时间，不保证是媒体的准确发布时间；标题保持来源语言。使用或再分发 GDELT 数据时须注明并链接 [GDELT Project](https://www.gdeltproject.org/about.html)。

机场运行必须区分“当前快照”和“适用于目标起飞时刻的模型信号”。FAA NAS Status 是美国当前运行事件的权威来源，但机场没有出现在事件列表中只表示没有列出的当前 FAA 事件；不能推导出所有航班正常。范围有限的 `freeForm` 限制不能当作完整机场关闭。FAA 事件只有在时间范围覆盖计划起飞时才可进入目标信号。

SerpApi Google Flights 首次搜索只产生候选。系统分别搜索经济舱、超级经济舱、商务舱和头等舱，并启用 `deep_search=true` 与 `show_hidden=true`；这些参数只能扩大可见范围，仍不能保证返回所有航司、航班、舱位、销售方、私有票价或日期。候选必须带 `booking_token`，随后查询到的 `selected_flights` 必须与原始航段完全一致，并至少返回一个具有销售方、匹配航班号、正数 USD 价格和 HTTPS `booking_request.url` 的购票选项，才能标为 `booking_option_confirmed`。这证明 Google Flights 在来源结果生成时点返回了对应购票路径，不等于航空公司或销售方最终结账页仍有相同库存、规则或价格。若该请求无需 POST，`booking_url_kind=direct_get` 并可打开 Google 的销售方跳转；若请求带 `post_data`，普通 `<a>` 不能重放它，系统明确返回 `booking_url_kind=google_flights_itinerary` 的 Google Flights 结果页，不把它误称为销售方直链或“已选行程”。官方在此只保证 `google_flights_url`。当前响应不能可靠证明报价是否含税，因此 `taxes_included` 保持未知，不能写成“明确含税”。

[SerpApi 免费计划](https://serpapi.com/pricing) 目前提供每个 provider 结算周期 250 次成功搜索及每小时 50 次请求；周期按账户 `plan_renewal_date` 划分，不按自然月。四舱初始搜索和 `booking_token` 查询共用额度。SerpApi、SearchAPI 与显式发布后的 Ignav 都会为其搜索实际返回的全部合格候选请求二次验证；不存在应用定义的每次比较候选数量上限。原子账本只批准实际剩余额度覆盖的调用，额度不足的候选可与已验证报价一同返回，但覆盖必须标为 `partial` / `quota_limited`，不能解释为没有航班。若 SerpApi 返回 `Processing/Queued`，系统只对白名单格式的同一 Search ID 做 0.5、1、1.5、2 秒有界 Archive 轮询；有界轮询仍未完成、供应商搜索错误、传输错误或 HTTP 408/425/5xx 时，只有再次原子预留额度成功后才允许受控重提一次，第二次可以轮询但不会产生第三次提交。HTTP 400、认证失败、429 与严格解析拒绝不重试；重试因额度失败由 `retry_quota_limited` 与覆盖字段报告。候选数量随查询变化，验证调用只受实际免费额度和供应商响应约束；额度不足时允许返回已验证部分，并通过 `coverage_scope`、各候选计数、`coverage_status` 和 `quota_limit` 明确标记截断。官方当前说明缓存、错误和失败搜索不计入 provider 周期额度，但项目本地账本会在每次尝试前保守预留，连 provider 缓存请求也预留，因此本地 `monthly_calls_used` 是尝试数而不是 provider 成功计费数，可能更高并提前停止。`SERPAPI_MONTHLY_LIMIT` 和保存在忽略 Git 的 `runtime/` 目录中的单一结算周期账本共同执行硬上限：省略时默认 250，大于 250 时钳制为 250，非法或非正值也不会解除上限。部署者仍应独立检查账户用量。

`auto` 不会在第一个确认报价后停止：所有已配置且获准参与的严格源都会运行并各自消耗自己的免费额度。若至少两个来源实际运行，顶层 metadata 使用 `provider_code=strict_fare_aggregate`，并以 `provider_runs` 保留逐源状态、计数、额度及脱敏诊断；聚合层只接收各来源已独立二次验证的报价。跨销售方、跨来源的去重键是完整有序航段身份与舱位，同组仅保留最低最终确认价格并保留获胜来源证据。单源结果继续使用该来源自己的 provider code。多个免费源可以减弱单一来源覆盖或额度不足的影响，但不能绕过各自额度墙，也不能证明全球全部可售库存；“严格”表示证据不降级，不表示覆盖全球所有航司、舱位、销售方或日期。

SerpApi 可返回最长约 1 小时的 provider 缓存；应用另有 5 分钟严格结果缓存。`provider_cache_hit` 来自本地缓存命中或 provider 状态/时间启发式，只表示检测到缓存复用，不精确区分或证明具体来源；`provider_cache_age_seconds` 是当前响应相对 `verified_at`（SerpApi `search_metadata.created_at`，即 provider 结果生成时间）的年龄，并限制在含处理/时钟容差的 0–3900 秒。主页另外显示本次响应生成时间，不能把它与 provider 结果时间混为一谈。

AirLabs 免费 `schedules` 是近实时接口，结果按当前或目标时刻前后 90 分钟筛选，并受最多 50 行、免费配额和提供方可见计划时段限制，因此是实际航班样本而不是完整机场统计。航班比较的 schedules/routes 查询也各自最多 50 行；任一响应的 `request.has_more` 为真或达到 50 行时，`schedule_sample_truncated=true`，表示参考样本可能不完整。本演示为控制免费配额和调用量不继续分页。`routes` 表达按星期重复的周期时刻表；系统可以把完整记录投影到所选日期，但必须标为 `recurring_timetable_projection`，不能称为当天已确认执行或可售航班。两类 AirLabs 记录都只进入 `timetable_references`。所有 AirLabs 传输在同一个跨进程 SQLite 账本中预留；没有有效的 `AIRLABS_MONTHLY_CALL_LIMIT` 时 fail closed，provider 返回的账户月上限和已用量只能进一步收紧本地硬墙。

比较请求只提交 `departure_date`，且日期不得早于出发机场当地今天、并在当地今天后 370 天以内。未来日期使用当地正午作为模型、天气和新闻参考；同日若正午仍有超过 30 分钟余量则使用正午，否则沿 UTC 时间线推进 30 分钟再转换回机场当地时间。安全参考跨入次日则返回 422。`departure_time_basis` 明确区分正午与剩余同日参考，两者都不是航班钟点。严格主列表只接受完整、连续、未来、同一真实舱位且通过当前来源购票选项二次验证的 1–4 个航段。非 USD/非正价格、混合舱位、技术停靠、断裂航段、航班号不一致或二次验证失败都被拒绝；所有配置并获准的严格源都查询完成后再聚合，最终无结果也不生成模型航班。AirLabs live/routes 仍只进入不参与排名的 `timetable_references`。普通免费报价查询不能验证实际学生专属折扣；该字段显示 `unknown`，学生排序在没有报价级证据时不给该条件加分。

主页每个严格 live offer 都链接到 `GET /details/offer`，并通过 `POST /v1/offer-detail` 获取完整日期级 schedule、模型结果与从当前日期到起飞前的逐日价格模型预测曲线；周期参考没有详情链接。天气和新闻详情页继续使用各自的上下文接口。

天气与新闻详情页从主页接收 `departure_date`，每次刷新由服务重新计算带 `departure_time_basis` 的模型/上下文参考，避免沿用已过期的同日固定时刻；这些时间不是航班计划。天气详情接口在出发机场参考时刻和到达机场模型估算抵达参考分别查询 Open-Meteo，并显示当前条件、目标小时、前后 12 小时趋势和风险拆解；可用时还显示 NOAA METAR/TAF 原始报文与自动解释。新闻详情接口最多显示最近 7 天的 20 篇匹配文章及分类、命中词和时效权重。网页分别每 10 分钟和 15 分钟刷新，但服务端同样使用短期缓存以遵守免费来源的负载与配额边界。

Offer 详情初次加载发送 `force_refresh=false`，复用 5 分钟严格缓存而不主动发起 provider 请求；只有用户点击“刷新并重新查询”才发送 `force_refresh=true`，重新处理四舱搜索实际返回的全部合格候选，仍受实际免费额度与供应商响应限制。详情响应沿用 `weather_feature_status`，在准点预测切换到无天气模型时明确提示天气变量已忽略。主页和刷新详情请求最长等待 10 分钟；超时后保留错误信息，避免页面无限显示“正在比较”。

详细的状态、缓存、严格政策字段和失败回退语义见 [`runtime-context.md`](runtime-context.md)。免费服务的配额、覆盖和条款可能变化，公开部署前应重新核对官方说明。

### 配置严格航班搜索

在启动服务的同一个 PowerShell 窗口设置；不要把真实 key 写入文档或提交到 Git：

```powershell
$env:FLIGHT_OFFER_PROVIDER="auto"
$env:SERPAPI_API_KEY="your-serpapi-key"
$env:SERPAPI_MONTHLY_LIMIT="250"
$env:SEARCHAPI_API_KEY="your-searchapi-key"
$env:SEARCHAPI_LIFETIME_LIMIT="100"
python -m flight_forecaster serve --model-dir artifacts/demo
```

### 配置可选的免费 AirLabs key

在启动服务的同一个 PowerShell 窗口设置：

```powershell
$env:AIRLABS_API_KEY="your-free-key"
$env:AIRLABS_MONTHLY_CALL_LIMIT="1000"
$env:AIRLABS_USAGE_DB="runtime/airlabs-usage.sqlite3"
python -m flight_forecaster serve --model-dir artifacts/demo
```

应用读取进程环境变量，不会自动加载 `.env`。不要把 key 写入源码、浏览器/前端、应用日志或提交到 GitHub。服务端只把凭据发送给对应的 HTTPS provider，因此日志和错误信息不得记录完整外部请求 URL。未配置 AirLabs key 是受支持的上下文运行方式；配置 key 却没有有效的 `AIRLABS_MONTHLY_CALL_LIMIT` 时会在网络调用前拒绝，相关机场运行信号使用明确标记的 `model_fallback`，不会伪造 provider 数据。严格报价是否可用取决于至少一个已配置且有剩余额度的严格来源。

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
