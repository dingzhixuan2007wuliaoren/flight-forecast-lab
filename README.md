# Flight Forecast Lab / 全球航班预测实验室

Flight Forecast Lab 是一个可复现的双任务机器学习项目，用于比较未来行程的**模型票价**与**准点概率**。主页只要求输入出发机场、到达机场和出发日期；距离、飞行时长、天气、机场运行压力和近期新闻均由系统自动解析。

Flight Forecast Lab is a reproducible dual-task project with one fare model and two on-time variants for comparing **estimated fares** and **on-time probabilities**. The dashboard only asks for origin, destination, and departure date; distance, duration, weather, airport operations, and recent news are resolved automatically.

> 重要：默认票价预测与准点率模型使用确定性的合成演示数据训练。配置受支持的严格报价源（SerpApi、SearchAPI.io，或完成双开关显式发布后的 Ignav）后，主价格可显示经购票选项二次验证的来源结果报价；该结果可能使用 provider 缓存。模型估价、价格曲线和准点率仍不是经过全球真实数据验证的生产预测。
>
> Important: the default fare and on-time models use deterministic synthetic demo data. With a supported strict fare source configured (SerpApi, SearchAPI.io, or Ignav only after its two explicit release switches), the primary price can be a provider-result fare that also passed booking-option verification; that result may use provider caching. The model estimate, price curve, and on-time probability remain unvalidated demo predictions.

## 功能 / Features

- 页面右上角 `中文 / English` 切换，并使用浏览器本地存储记住语言。
- 比较页面只输入日期；服务在内部使用出发机场当地正午或仍安全的同日 +30 分钟时刻作为模型、天气和新闻参考，以 `departure_time_basis` 明确标记，且不把它显示成实际航班钟点。
- 119 个内置全球主要机场；其他有效 IATA 机场可通过免费的 OurAirports 目录回退解析。
- 内置 60 家主要航司的保守政策目录；严格主列表只显示二次验证实际返回的航司与舱位，未知航司也不会被默认扩展成四种舱位。
- 对活动严格报价源四舱搜索实际返回、并在免费额度与供应商响应范围内成功严格验证的候选提供三类排序：直飞优先、低价优先、学生友好优先；它们仍不是所有航司或全球航班的全量清单。
- 学生友好排序按：最低价格 → 已确认免费托运行李 → 已确认实际学生折扣 → 已确认免费改签/退票 → 年龄与验证要求。普通 Google Flights 免费查询不能确认实际学生专属折扣，因此该项保持 `unknown` 且不加分。
- 无需密钥的 Open-Meteo 当前天气/小时预报与 NOAA METAR/TAF 航空气象；美国机场使用 FAA NAS Status 当前运行事件，其他机场可使用 AirLabs 航班样本或 ADSB.lol 飞机密度代理；新闻来自无需密钥的 GDELT。
- 天气和新闻卡片以及每个严格确认报价均可进入独立的中英双语详情页；offer 详情含真实返回航段、转机等待时间与逐日价格模型预测曲线，天气和新闻页支持手动刷新，并分别每 10 分钟、15 分钟自动刷新。
- 外部服务超时、无数据或额度不足时，界面会明确标记回退或截断。票价模型不使用天气；准点率仅在天气状态为 `live` 或 `forecast` 时使用含天气模型，其他状态自动切换为无天气模型并显示“本次准点预测已忽略天气变量”。
- FastAPI、OpenAPI 文档、CLI 训练入口、时间切分评估和自动测试。

The UI always labels external context as `live`, `forecast`, `proxy`, `historical`, or `neutral`. Missing policy information remains `unknown`; the service never converts “unknown” into “not included.”

## 免费数据模式 / Strict-free data mode

| 数据 | 默认来源 | 无法获取时 |
| --- | --- | --- |
| 天气 | 2 小时内使用 Open-Meteo 当前模型天气并结合 NOAA METAR/TAF；2–30 小时结合小时预报与 TAF；其后至 16 天使用 Open-Meteo 小时预报 | 可展示明确标记的训练同月平均值/季节先验，但准点预测切换到无天气模型，不把 proxy 当作天气输入 |
| 机场运行 | 美国机场优先使用无需密钥的 FAA NAS Status 当前事件；其他机场配置 AirLabs 免费密钥后使用计划出港样本，否则显示 ADSB.lol 当前飞机密度代理 | 与目标起飞时刻不匹配时，模型使用明确标记的训练平均值/合成先验；当前快照仍单独展示 |
| 时事新闻 | 无需密钥的 GDELT DOC 2.0 最近 7 天中断新闻，按最新观察时间排序；DOC 不可用时使用官方 GAL 滚动 RSS | 先使用最多 6 小时的带标签缓存并降低影响；没有缓存时返回中性值且不生成新闻 |
| 严格航班比较 | `auto` 查询每个已配置且获准参与的严格报价源；SerpApi、SearchAPI.io 以及双开关显式发布后的 `ignav_verified_fares` 都必须独立二次验证完整航段、正数票价与安全 HTTPS 路径 | 两个或更多严格源实际运行时返回 `strict_fare_aggregate`，并在 `provider_runs` 保留逐源状态；同一完整航段与舱位跨源只保留最低已验证价格。全部来源都没有可验证报价时才返回带原因的结构化空 `offers`；参考源绝不补位 |

AirLabs 密钥是可选项。没有任何密钥时项目仍可运行。Open-Meteo 公共免费端点限非商业用途并采用 CC BY 4.0；GDELT 允许免费使用，但使用或再分发其数据时须注明并链接 GDELT。免费额度、覆盖和条款可能变化，公开部署前应重新检查 [Open-Meteo 官方条款](https://open-meteo.com/en/terms)、[GDELT 项目说明](https://www.gdeltproject.org/about.html)与其他来源要求，详见 [运行时数据与回退](docs/runtime-context.md)。

机场运行采用两层语义：`context.operations` 顶层是实际送入目标起飞时刻预测的信号，`context.operations.current_snapshot` 是查询时刻的机场快照。两者可能不同。例如，数周后的航班会使用训练平均值/合成先验进行预测，同时仍可展示今天的 FAA 事件或飞机密度。FAA 的 `freeForm` 限制（例如只影响部分通航飞机）只作为低权重当前提示，不能被解释为整个商业机场关闭；ADSB.lol 也只表示机场附近飞机密度，不提供真实延误、取消或地面停飞数据。

Airport operations use two distinct layers: the top-level `context.operations` is the target-departure signal actually used by the model, while `context.operations.current_snapshot` describes conditions at request time. FAA scoped restrictions are not treated as full commercial-airport closures, and ADSB.lol aircraft density is only a traffic proxy—not an actual delay, cancellation, or ground-stop feed.

Open-Meteo 的“当前天气”来自约 15 分钟分辨率的天气模型，并非机场传感器实测；NOAA METAR 才是机场观测。页面会分别标记 `open_meteo_current_model`、`noaa_metar`、`forecast`、`live` 或 `proxy`，不会把远期历史平均值冒充实时天气。

Open-Meteo current conditions are model-derived at roughly 15-minute resolution, while NOAA METAR provides airport observations. Beyond the free forecast horizon, the labelled training-average fallback is display-only and the on-time prediction switches to the no-weather model rather than using today's weather.

天气不进入票价模型。准点率预测只在适用于目标出发时刻的天气状态为 `live` 或 `forecast` 时选择含天气模型；`proxy`、`historical`、`neutral` 或 `unavailable` 时选择单独训练的无天气模型。页面仍可展示回退天气作为参考，但会返回 `weather_feature_status=ignored` 并明确提示本次准点预测已忽略天气变量。

Weather is not a fare-model feature. The on-time forecast selects the weather-aware model only for an applicable `live` or `forecast` signal; `proxy`, `historical`, `neutral`, and `unavailable` select the separately trained weather-free model. A fallback weather card may remain visible for context, but `weather_feature_status=ignored` explicitly says weather was omitted from that on-time prediction.

For a date-only comparison, weather is resolved once at the documented origin-local reference time. An individual verified flight uses that weather only when its actual departure is within two hours of the reference; flights farther away switch to the no-weather model and carry `offers[].weather_feature_status=ignored` in the table and detail page.

## 快速运行 / Quick start

支持 Python 3.11–3.13。Windows PowerShell：

```powershell
git clone https://github.com/dingzhixuan2007wuliaoren/flight-forecast-lab.git
cd flight-forecast-lab

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

训练并启动：

```powershell
python -m flight_forecaster train-demo --output artifacts/demo
python -m flight_forecaster serve --model-dir artifacts/demo
```

打开：

- 双语页面 / bilingual dashboard: <http://127.0.0.1:8000/>
- API 文档 / API docs: <http://127.0.0.1:8000/docs>
- 健康检查 / health: <http://127.0.0.1:8000/health>

先在主页完成一次比较，再点击天气、新闻卡片或任一 offer 的“查看详情 / View details”。详情链接在当前标签页打开并可复制分享；返回主页时，浏览器会话中的表单、排序和最近一次比较结果会恢复。

macOS 或 Linux 激活环境：

```bash
source .venv/bin/activate
```

### 严格未来报价与免费额度 / Strict future fares and free quota

所有外部调用都在本地免费硬上限前预留额度；达到上限即停止，不自动购买、不透支，也不把参考数据升级成可售航班。`GET /v1/provider-status` 和主页状态面板只返回配置布尔值、角色、脱敏额度状态及双语说明，不返回密钥、原始错误或完整外部 URL。

Every external call reserves free allowance before network I/O. The provider stops at the local hard ceiling—there is no automatic purchase or overage, and reference data is never promoted into a bookable flight. `GET /v1/provider-status` and the dashboard expose only configuration booleans, roles, sanitized quota state, and bilingual notices; they never expose credentials, raw provider errors, or complete outbound URLs.

`/v1/provider-status` 是只读的本地脱敏账本：访问端点、刷新主页或刷新提供商二级页都不会请求外部提供商，因此不消耗其查询额度。每个额度数字都必须标明 `quota_data_basis`：`provider_reported` 是先前业务响应或受控同步已取得的供应商脱敏快照；`local_ledger` 是本地保守预留账本，不是供应商账户余额；`provider_and_local_ledger` 是两者的保守组合，不是相加。没有可验证数据时保持“未知”，不猜测为零或满额。

`/v1/provider-status` is a read-only sanitized local ledger. Loading the endpoint or refreshing either status view makes no external-provider request and consumes no provider allowance. Every quota number carries a `quota_data_basis`: `provider_reported` is a sanitized snapshot already captured from an earlier business response or controlled sync; `local_ledger` is a conservative local reservation ledger, not the provider account balance; and `provider_and_local_ledger` is a conservative combination, never a sum. Unverifiable usage remains unknown rather than being guessed as zero or full.

小时级或供应商特定的临时限流以 `temporarily_rate_limited=true` 单独显示；它不会把仍有余额的结算周期或终身额度改写成零。只有与该额度窗口一致的耗尽证据，才会显示为额度耗尽。

Hourly or provider-specific throttling is exposed separately as `temporarily_rate_limited=true`; it never rewrites a still-positive billing-period or lifetime balance to zero. A provider is shown as quota-exhausted only when the evidence applies to that same quota window.

| 来源 / Source | 严格角色 / Strict role | 免费硬上限语义 / Free hard-stop semantics |
| --- | --- | --- |
| SerpApi · Google Flights | 首个配置顺序的可聚合严格源 / first configured aggregating strict source | 每个 provider 结算周期最多 250 次本地预留请求；不是自然月 / up to 250 locally reserved requests per provider billing period, not calendar month |
| SearchAPI.io · Google Flights | 可聚合严格报价源 / aggregating strict fare source | 注册账户一次性 100 次请求；不会按月恢复 / one-time 100-request signup allocation; no monthly renewal |
| Ignav | 默认 `ignav_quarantine`；显式发布后为 `ignav_verified_fares` / quarantined by default; separately identified after release | 一次性最多 1,000 次请求；只有 `IGNAV_STRICT_RELEASE=1` 与 `IGNAV_FREE_ACCOUNT_ATTESTED=1` 同时成立且完成受控验证才可进入主列表 / up to 1,000 lifetime requests; both explicit switches and controlled validation are required |
| Scrape.do · Google Flights | 仅聚合覆盖参考 / aggregate coverage reference only | 每月 1,000 点数，每次参考调用预留 10 点数，瞬时失败最多受控重试一次 / 1,000 credits per month, 10 reserved per reference call, at most one controlled transient retry |
| OpenSky Network | 当前航迹密度参考 / current trajectory-density reference | 无凭据时匿名模式默认启用，每日 400 API 点数；不能证明票价或库存 / anonymous access is active without credentials at 400 API credits per day; no fare or inventory proof |
| AeroDataBox | 日期级时刻参考 / dated schedule reference | 优先采用可信 RapidAPI 免费计划账期头；没有可信重置信号时使用安装生命周期 600 单位硬墙，绝不按自然月自行重置 / trusted RapidAPI free-plan reset headers define a cycle; otherwise a 600-unit installation-lifetime wall applies |
| AirLabs | 机场运行与时刻参考 / operations and timetable reference | 所有进程和调用点共用 SQLite 月度账本；缺少明确本地上限时拒绝调用，响应中的账户月上限/已用量只能进一步收紧硬墙 / one cross-process SQLite ledger; missing local limit fails closed and provider counters only tighten the wall |

额度快照的更新策略也是免费优先：[SerpApi Account API](https://serpapi.com/account-api) 官方明确说明免费且不计入月度额度，可由独立、受控的后端流程刷新，但当前状态页不会调用它。SearchAPI.io 的账户端点与 Scrape.do 的 `/info` 虽公开了用量字段，但官方未说明查询免额度，因此系统不自动调用。AeroDataBox/RapidAPI 和 OpenSky 只在正常业务请求的响应头中采集余额信号，不发出额外探测请求。AirLabs 只读本地账本和先前有效响应中已有的脱敏计数。

Quota snapshots follow the same free-first policy. The official [SerpApi Account API](https://serpapi.com/account-api) is free and excluded from monthly quota, so a separate controlled backend workflow may refresh it, but the current status views do not call it. SearchAPI.io's account endpoint and Scrape.do's `/info` publish usage fields without documenting whether those checks consume allowance, so they are not queried automatically. AeroDataBox/RapidAPI and OpenSky quota headers are captured only from normal business responses—never from an extra probe. AirLabs uses its local ledger and sanitized counters already present in prior valid responses.

主比较会查询每个已配置且获准参与的严格报价源，并使用每个候选返回的二次验证令牌/标识查询购票选项。只有二次响应中的已选航段与原候选完全一致，且存在带 HTTPS
购票请求、销售方、匹配航班号、完整航段、真实舱位和正数 USD 价格的结果才会进入 `offers`。
若销售方动作是可直接打开的 GET 跳转，`booking_url_kind=direct_get`；Google 常见的
`booking_request` 需要 POST 时，系统不丢弃 POST 数据后伪装成直链，而是提供
`booking_url_kind=google_flights_itinerary` 的 Google Flights 结果页，由用户继续选择销售方；官方只保证返回 `google_flights_url`，因此不能把它写成“已选行程”或销售方直链。

SerpApi 免费计划目前包含每个 provider 结算周期 250 次成功搜索及每小时 50 次请求；周期边界
以账户返回的 `plan_renewal_date` 为准，不按自然月切换。初始四舱搜索与 `booking_token` 后续查询
共享这项额度。系统解析四个舱位搜索实际返回的合格候选，但 SerpApi 每次比较最多二次验证 6 个；
超出的候选标为 `partial` / `quota_limited`，同一航程和舱位有多个已验证销售方时只保留最低价。`deep_search=true` 与
`show_hidden=true` 会扩大 Google Flights 可见候选，但仍不能保证所有航司、航班、舱位、销售方
或私有票价都被返回。额度不足时，已成功验证的结果仍可显示，未验证候选会被跳过，并通过
`fare_search_metadata.coverage_status`、计数字段、`quota_limit` 和中英双语提示明确标记截断；
系统不会使用模型航班补位：

```powershell
$env:FLIGHT_OFFER_PROVIDER="auto"
$env:SERPAPI_API_KEY="your-serpapi-key"
$env:SERPAPI_MONTHLY_LIMIT="250"
$env:SEARCHAPI_API_KEY="your-searchapi-key"
$env:SEARCHAPI_LIFETIME_LIMIT="100"
python -m flight_forecaster serve --model-dir artifacts/demo
```

`auto` 会并发运行所有已配置且获准参与的严格源，而不是在第一个成功来源后停止。只有一个来源实际运行时保留该来源自己的 provider code；两个或更多来源实际运行时，顶层 `fare_search_metadata.provider_code=strict_fare_aggregate`，`provider_runs` 保存每个来源的独立状态、计数、额度与诊断。SerpApi、SearchAPI 和显式发布后的 Ignav 各自完成四舱搜索后，每个来源、每次比较最多二次验证六个候选；候选顺序先覆盖每个有结果的舱位，再按供应商搜索价格处理。未验证部分明确标记为 `partial` / `quota_limited`，不伪装成“没有航班”。SearchAPI 的四舱搜索并发执行，而且只为该有界验证批次预留额度，因此中断不会再一次占满全部剩余额度。聚合只合并各来源已经独立通过二次购票验证的报价，并按“完整航段序列 + 舱位”跨源去重，保留最低最终确认价。任一来源失败、处理中或额度受限都会保留在逐源状态和聚合覆盖状态中；只有全部已完成来源都没有可验证报价时才返回结构化空结果。SearchAPI 的本地硬墙是安装生命周期 100 次，不按月重置，也不应配置成月额度。每个来源都会消耗自己的免费额度，因此增加来源可以缓解单源覆盖不足，但不能把多个免费计划变成无限额度，也不保证全球全量航班。前端会显示比较耗时，30 秒后说明仍在验证供应商，并在 150 秒停止浏览器等待且保留可见错误；重复点击不会创建第二个并行请求。

若 SerpApi 返回 `Processing` 或 `Queued`，后端会按 0.5、1、1.5、2 秒的有界退避读取固定的
Search Archive 地址；它只轮询同一个经过白名单校验的 Search ID，不跟随响应中的任意 URL，
Archive 读取单独记录为 `archive_poll_count`，不计入 `call_count` 或本地额度预留。若有界轮询后仍在处理，或发生供应商搜索错误、传输错误、HTTP 408/425/5xx，系统只有在再次原子预留免费额度成功后才受控重提一次；第二次提交可再次有界轮询，但绝不会触发第三次提交。HTTP 400、认证失败、429 或严格解析拒绝不会重试。最终仍在处理返回 `provider_processing`，终态/HTTP/网络错误返回 `provider_error`，只有成功完成但没有通过严格验证的购票选项才返回 `no_results`；重试额度不足由 `retry_quota_limited`、`coverage_status` 与 `quota_limit` 如实标记。

When SerpApi reports `Processing` or `Queued`, the backend performs bounded polling of the same
allowlisted Search ID through the fixed Search Archive endpoint. It never follows an arbitrary
provider URL. After an unresolved poll or another approved transient failure, it may resubmit once
only after a fresh atomic quota reservation; the second submission can poll but cannot cause a
third submission. HTTP 400, authentication failures, 429 responses, and strict-parse rejections are
not retried. The API distinguishes `provider_processing`, `provider_error`, and a successfully
completed `no_results` response, and exposes a retry blocked by quota without pretending the first
attempt was skipped.

异常诊断只保留观测时间、阶段、HTTP 状态、固定异常类型和经过格式校验的 Search ID，并写入
Git 已忽略的 `artifacts/runtime/serpapi-usage.sqlite3`，最多保留 500 条；最近最多 10 条也会作为
`fare_search_metadata.diagnostics` 返回。API Key、`booking_token`、请求参数、完整 URL 和原始错误
文本均不会保存。

`SERPAPI_MONTHLY_LIMIT` 可省略（默认 250）；大于 250 会被钳制为 250，非法或非正值也不会
解除安全上限。凭据只放在启动服务进程的环境变量或本地密钥存储中，绝不能提交到 GitHub、
发送给浏览器/前端或写入应用日志。服务端按 SerpApi 官方接口要求把 key 放入发往
`https://serpapi.com` 的 HTTPS 查询参数；因此不要记录完整外部请求 URL。免费额度和条款可能变化，请在使用前复核
[SerpApi pricing](https://serpapi.com/pricing) 与
[Google Flights API 文档](https://serpapi.com/google-flights-api)。Google Flights 及购票选项是
来源结果生成时点的快照；即使通过严格验证，也不保证航空公司或销售方结账页仍有相同库存、规则或最终价格。
“购票选项已验证”证明二次响应存在匹配购票路径，不表示页面链接一定是销售方 GET 直链。
本地 `monthly_calls_used` 是按 `plan_renewal_date` 周期保守预留的尝试数，不是 SerpApi
成功计费数。即使 provider 缓存请求按官方规则免费，本地仍会先预留，因此可能提前停止，
以优先保证不会超过免费限额。
SerpApi 可能返回最长约 1 小时的 provider 缓存。响应以 `provider_cache_hit` 和
`provider_cache_age_seconds` 显示当前结果是否被缓存复用及其相对 `verified_at` 的年龄；
`verified_at` 是 provider 结果生成时间，不是本次浏览器请求时间。缓存命中可能来自 SerpApi
一小时缓存或应用的 5 分钟严格缓存；该布尔值来自本地命中或 provider 状态/时间启发式，
字段不证明具体缓存来源。价格中的税费是否已包含无法从当前
免费响应可靠确认，固定显示为未知，付款前必须核对。

### 可选免费 AirLabs 密钥 / Optional free AirLabs key

免费 AirLabs 密钥用于非美国机场运行样本和独立的日期/周期时刻参考。`schedules` 是近实时计划接口，免费层通常只覆盖临近当前时刻的有限窗口；`routes` 是周期时刻表，不是所选日期已经确认运营的航班。两者都只进入不参与排名的 `timetable_references`，不能单独创建严格 offer。AirLabs 不提供座位库存或可售报价。所有 AirLabs 传输在网络调用前通过同一个跨进程 SQLite 账本预留一次调用；设置 key 却没有有效的 `AIRLABS_MONTHLY_CALL_LIMIT` 时会 fail closed。有效响应中的 `request.key.limits_by_month` 与 `limits_total` 会保存为脱敏账户快照，并且只能收紧后续可用量。

没有 AirLabs 密钥时，美国机场仍可使用 FAA，其他机场运行页显示 OpenSky/ADSB.lol 代理并在需要时使用模型先验；它不会影响独立的严格报价验证。若没有任何可用严格报价源，严格比较返回结构化空结果，不再用纯模型航班填充列表。密钥只保存在本地环境变量，不要提交到 GitHub：

```powershell
$env:AIRLABS_API_KEY="your-free-key"
$env:AIRLABS_MONTHLY_CALL_LIMIT="1000"
$env:AIRLABS_USAGE_DB="runtime/airlabs-usage.sqlite3"
python -m flight_forecaster serve --model-dir artifacts/demo
```

完全离线模式：

```powershell
$env:EXTERNAL_CONTEXT_ENABLED="0"
python -m flight_forecaster serve --model-dir artifacts/demo
```

可参考 [.env.example](.env.example)。当前启动命令直接读取进程环境变量，不会自动加载 `.env`；请在启动服务的同一个终端设置变量，并且不要提交密钥文件。

## 比较接口 / Comparison API

`POST /v1/compare`

```json
{
  "origin": "YYZ",
  "destination": "LHR",
  "departure_date": "2026-09-15"
}
```

响应包括：

- 自动推算的距离和飞行时长；
- 天气、机场运行和新闻信号及来源、状态、时间；机场运行还区分目标时刻信号与 `current_snapshot` 当前快照；
- 仅针对活动严格源搜索后又通过该来源购票选项二次验证的未来报价，显示来源结果价格、缓存状态与年龄、真实返回舱位、完整航段、模型价格区间、准点率和风险；
- 行李、学生计划、退改、年龄及验证的保守状态；
- `direct_first`、`lowest_price`、`student_first` 三组覆盖本次返回的全部严格验证 offer ID；
- `availability_mode=strict_bookable_only`、`fare_search_metadata`、结构化 `result_status`、双语 `strict_mode_notice`，以及不参与比较的 `timetable_references`；两个或更多严格源实际运行时，metadata 使用 `strict_fare_aggregate` 并以 `provider_runs` 保留每个来源的状态和额度；顶层还以 `coverage_scope=provider_returned_booking_verification_candidates`、`eligible_candidate_count`、`verification_attempted_count`、`verified_candidate_count`、`strictly_rejected_candidate_count`、`provider_failed_candidate_count`、`search_failed_cabin_count`、`quota_skipped_candidate_count`、`deduplicated_verified_count`、`coverage_status`、`quota_limit` 与 `retry_quota_limited` 解释跨源候选覆盖和截断；某个舱位搜索失败时，其他舱位已独立严格验证的报价仍可保留，同时覆盖状态明确标为不完整；若 `cache_hit=true`，调用计数为本次新增调用（零），覆盖统计则明确来自原查询；
- `departure_date`、内部参考时刻及 `departure_time_basis`；
- 免费航班查询是否触及 50 行上限的 `schedule_sample_truncated` 与 `schedule_sample_limit`；
- 中英双语限制说明。

The endpoint defaults to strict bookable-offer mode. In `auto`, every configured and released strict
source runs; `offers` contains only results that survive that source's own second booking-option request
with an identical selected-flight response and a usable booking option. With two or more attempted
sources, metadata uses `strict_fare_aggregate` and preserves each source in `provider_runs`. AirLabs
schedules/routes, route-level hints, catalogue-expanded cabins, and
model-only flights never enter `offers` or rankings. The verified provider-result fare remains separate
from the synthetic model estimate and forecast curve, and it is not a final-checkout guarantee.
For each comparison, SerpApi, SearchAPI, and released Ignav each attempt booking-option verification
for at most six eligible candidates returned by their four cabin searches. Excess candidates are
reported as `partial` / `quota_limited`, rather than as no flights.
The response keeps only the lowest verified price across sellers and sources for the same complete
segment sequence and cabin. Coverage
counters and `coverage_status` distinguish complete candidate processing from quota or provider
truncation. Even with `deep_search` and `show_hidden`, results are not a complete global inventory.
Tax inclusion is unknown.

比较接口要求 `departure_date` 不早于出发机场当地今天，且不超过当地今天后 370 天。未来日期使用当地正午作为模型和上下文参考；同日查询若正午仍比生成时刻晚超过 30 分钟也使用正午，否则使用沿绝对时间线推进 30 分钟的同日参考。若该安全参考已跨入次日则返回 422。响应分别以 `departure_time_basis=origin_local_noon_model_reference` 或 `origin_local_remaining_day_model_reference` 标注；两种参考都不是实际航班钟点。实际航班钟点来自确认报价的逐段行程。AirLabs 免费查询最多取 50 行且只作为参考；`schedule_sample_truncated=false` 不能证明覆盖完整。

## 单项接口 / Individual endpoints

模型票价：`POST /v1/predict/price`

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "airline": "DL",
  "cabin": "economy",
  "stops": 0,
  "departure_time": "2026-09-15T08:00:00-04:00"
}
```

准点概率：`POST /v1/predict/on-time`

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "airline": "DL",
  "scheduled_departure": "2026-09-15T08:00:00-04:00"
}
```

天气严重度、出发机场拥堵、距离和飞行时长均不再由调用方提交。

## 详情页面与接口 / Detail pages and APIs

详情页面由主页生成带查询参数的可分享链接，并在当前标签页打开：

- `GET /details/weather`：同时展示出发机场模型/天气参考时刻与到达机场模型估算参考时刻的当前天气、目标时刻预报、前后 12 小时趋势、风险拆解，以及 NOAA METAR/TAF 原始报文、独立可用性元数据和保守的中英双语解释；这些参考时刻不是航班计划。页面可手动刷新并每 10 分钟自动刷新。
- `GET /details/news`：展示最近 7 天最多 20 篇匹配报道的原始标题、来源、语言、GDELT 索引时间、风险类别、命中词和时效权重，并单独标明详情页文章风险与实际进入模型的 `model_signal`；页面可手动刷新并每 15 分钟自动刷新。标题不做机器翻译。
- `GET /details/offer`：展示所选严格模式 offer 的航司、舱位、日期级完整时刻、价格/准点模型结果及行程依据，并显示从出发机场当地今天到起飞日前最后安全时刻的逐日价格模型预测曲线。主页中的每个 offer 都有独立入口。

天气与新闻两页优先调用相同的日期级请求：

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "departure_date": "2026-09-15"
}
```

服务会在每次手动或自动刷新时重新生成仍安全的当地正午/同日参考，并在响应中返回 `departure_time_basis`；因此不会反复提交已经过期的固定参考时刻。兼容客户端仍可改为只提交带偏移或按出发机场当地时区解释的 `departure_time`，此时响应 basis 为 `legacy_input`。两字段必须且只能提供一个。

- `POST /v1/context/weather-detail`
- `POST /v1/context/news-detail`

单个 offer 详情调用 `POST /v1/offer-detail`：

```json
{
  "origin": "JFK",
  "destination": "LAX",
  "departure_date": "2026-09-15",
  "offer_id": "off_0123456789abcdef01234567"
}
```

严格模式详情只对应主列表中已通过当前来源购票选项二次验证的报价。每个航段的 `data_basis` 和顶层 `schedule_source` 必须与 `live_fare.provider_code` 一致（例如 SerpApi、SearchAPI 或显式发布后的 Ignav），不能硬编码成 SerpApi。首次打开详情以 `force_refresh=false` 复用 5 分钟严格缓存；只有页面“刷新并重新查询”按钮以 `force_refresh=true` 重新处理四舱搜索实际返回的候选，并继续受免费额度与供应商响应约束。已消失的报价不会由 AirLabs 或模型替代。`live_fare` 是来源结果生成时点的价格和舱位证据，可能使用 provider 缓存，且不保证航空公司或销售方最终结账价格或税费；`estimated_price_usd` 与 `price_curve` 仍是独立的演示模型输出。曲线每日改变模型的模拟查询时刻并保持同一行程、舱位、新闻快照和其他特征；它不是已采集历史票价或未来真实报价。详情页也会在 `weather_feature_status=ignored` 时明确说明本次准点预测已忽略天气变量。

The detail pages use the same `中文 / English` switch as the dashboard. Weather refreshes every 10 minutes and news every 15 minutes; both also provide a manual refresh button. Initial offer-detail load reuses the five-minute strict cache; only its refresh button forces a new four-cabin search and candidate verification, still bounded by free quota and provider responses. A provider result can itself be cached for up to about one hour, with cache status and age shown separately from response generation time. Offer detail explicitly reports when `weather_feature_status=ignored` and the on-time prediction used the weather-free model.

主页只显示当前启用来源的逐提供商剩余额度摘要；即使两个来源使用同名额度单位，也不会跨账户或跨重置周期相加。不同结算周期、账户终身额度和 API 点数不会被伪装成一个总数，未知用量也不会被猜测为满额或零。`/details/providers` 二级页面显示完整的脱敏逐来源状态、额度周期和严格报价边界；它只读取 `/v1/provider-status` 与同一浏览器会话中的最近快照，不会调用外部供应商或消耗查询额度。

The dashboard shows only a per-provider remaining-allowance summary for active sources. Even providers with the same named unit are not added across accounts or reset windows. Billing-period requests, lifetime allowances, and API credits are never presented as one total, and unknown usage is never guessed as full or zero. The `/details/providers` second-level page shows the full sanitized per-source status, quota window, and strict-fare boundary. It reads only `/v1/provider-status` and the latest same-session snapshot, so it makes no external provider call and consumes no provider allowance.

## 目的地景点、酒店与机场交通 / Destination places, hotels, and airport routes

完成一次航线比较后，严格航班列表上方会出现“目的地景点 / Attractions”和“目的地酒店 / Hotels”两个入口。列表页按真实数据标签分类：景点分为地标、博物馆、自然公园、娱乐和购物，住宿分为酒店、青年旅舍、民宿、汽车旅馆和公寓；每个列表最多返回 30 个有名称和坐标的结果。地点基础数据来自 OpenStreetMap，通过 Nominatim 确认机场服务城市，再以 5、15、30 公里渐进半径查询具名 OSM 节点，达到 30 条就停止扩大，避免对免费公共服务发出大查询。页面同时显示实际成功查询的半径和完整/部分覆盖状态；提前达到记录目标或较大半径暂时失败时，绝不会声称完成了 30 公里覆盖。一次列表操作中的主 Overpass 请求失败时，最多只会在官方公共实例列表中的全球 VK Maps 节点受控重试一次；成功结果缓存 24 小时，两个节点都失败时不会把故障缓存成“无结果”。来源缺少描述、星级、电话、开放时间、图片或官网时，页面明确显示“数据源未提供”，不会合成内容。

After a route comparison, two compact entries—Attractions and Hotels—appear above the strict-flight list. Their second-level pages classify real tagged places and return at most 30 named, geolocated results per list. OpenStreetMap supplies the place evidence: Nominatim resolves the airport's served city, then named OSM nodes are queried at progressive 5, 15, and 30 km radii, stopping once 30 results are available to keep public-service queries small. Each page reports the actual successfully queried radius and complete/partial coverage status; reaching the record target early or losing a wider-radius request is never presented as full 30 km coverage. A list operation makes at most one controlled retry on the currently listed global VK Maps public instance; successful responses are cached for 24 hours, while two-provider failures are not cached as false empty results. Missing descriptions, stars, contact details, opening hours, images, or websites remain unavailable rather than being invented.

点击任一地点会打开详情页，并按需查询从到达机场到该坐标的驾车、骑行和步行路线。路线来自 `routing.openstreetmap.de` 的全球 car/bike/foot OSRM 图；成功结果缓存 24 小时并遵守每秒最多一次请求，临时失败不缓存、下次可以重试。显示的分钟数是基于道路图的预计时间，不是实时交通承诺。全球公共交通需要当地 GTFS/NeTEx 时刻覆盖；当前版本在没有可验证时刻源时明确标为不可用，不用直线速度伪造公交或地铁时间。

Opening a place requests airport-to-place driving, cycling, and walking routes on demand. They use the worldwide car/bike/foot OSRM graphs at `routing.openstreetmap.de`; successful routes are cached for 24 hours and respect the public service's one-request-per-second policy, while transient failures are not cached. Durations are road-graph estimates, not live-traffic promises. Public transport requires local GTFS/NeTEx timetable coverage; without a verified source it is explicitly unavailable rather than estimated from straight-line speed.

酒店页的基础列表不会消耗付费或严格报价额度。只有用户填写入住日、退房日和成人数并点击“查询真实价格 / Check real prices”后，后端才会调用现有 SerpApi 账户的 [Google Hotels API](https://serpapi.com/google-hotels-api)。该请求与 Google Flights 严格报价共用同一个本地账本、供应商结算周期和小时限制；允许供应商缓存，只有本地一小时脱敏缓存命中不预留新额度。Processing/Queued 会使用经过格式验证的 Search ID 轮询固定归档地址；若仍未完成，最多只在再次原子预留额度后受控重提一次，并以一小时脱敏失败保护防止刷新或切换语言重复扣额。每晚价、总价、取消条件和销售来源只显示供应商同一报价记录实际返回的字段，税费包含情况未知，库存和价格仍可能变化。

The base hotel list consumes no fare-provider allowance. Only an explicit Check real prices action calls the existing SerpApi account's [Google Hotels API](https://serpapi.com/google-hotels-api) with the selected stay dates and adult count. Hotel searches share the exact same local ledger, provider billing window, and hourly cap as strict Google Flights searches. Provider caching is allowed, but only a sanitized local one-hour cache hit skips a new local quota reservation. Processing/Queued searches poll the fixed Search Archive URL for a validated Search ID and may be submitted only once more after another atomic quota reservation; a sanitized one-hour failure guard prevents refreshes and language changes from repeatedly spending calls. Nightly/total rates, cancellation terms, and seller attribution are shown only when they belong to the same returned quote; tax inclusion is unknown and availability can change.

## 新闻如何进入预测 / How news affects predictions

系统通过无需 API 密钥的 GDELT DOC 2.0 查询最近 7 天内与起点、终点及航空中断词相关的文章，并使用 `DateDesc` 按 GDELT 最新观察时间排序。DOC 请求失败时会尝试 GDELT 官方 GAL RSS：该 RSS 每分钟更新，保留最近约 15 分钟的文章。相同航线的成功结果缓存 15 分钟；实时来源暂时失败时可返回不超过 6 小时、明确标为 `historical` 且降低模型影响的旧缓存；没有可用缓存才返回值 0、状态 `neutral`，且绝不虚构标题。

仅对真实返回的标题进行保守评分，所得 `news_disruption_index` 是票价和准点模型的输入特征。主页卡片最多显示 5 篇去重后的来源文章，新闻详情页最多显示 20 篇。页面中的文章时间是 GDELT 观察 / 索引该文章的时间，并不保证等于媒体标注的准确发布时间；标题保留来源语言，系统不伪装成机器翻译标题。

The service queries the no-key GDELT DOC 2.0 API for route-related disruption coverage from the previous seven days and requests `DateDesc` ordering by GDELT's latest observed time. If DOC fails, it tries GDELT's official GAL RSS feed, which updates every minute and rolls over roughly the latest 15 minutes. A successful route result is fresh-cached for 15 minutes. When live providers fail, a cache no older than six hours may be returned as `historical` with reduced model influence; without a usable cache the signal is `neutral`, zero, and contains no invented headlines.

Only real returned titles are scored, and the bounded `news_disruption_index` feeds the fare model and the selected on-time variant. The dashboard card shows up to five deduplicated articles; the detail page shows up to 20. Article times are GDELT observed/indexed times and are not guaranteed to be exact publisher timestamps; headlines remain in their source language. The news relationship in the demo training data is synthetic. Article relevance does not prove that a particular flight will be affected, and the feature should be retrained and calibrated with lawful, representative historical snapshots before production use.

## 训练与验证 / Training and validation

```powershell
python -m pytest
python -m ruff check src tests
```

使用符合契约的真实 CSV：

```powershell
python -m flight_forecaster train-csv `
  --price-csv data/processed/price.csv `
  --ontime-csv data/processed/on_time.csv `
  --output artifacts/custom
```

详细字段见 [数据契约](docs/data-contracts.md)，真实训练来源见 [数据来源](docs/data-sources.md)，评估边界见 [模型卡](MODEL_CARD.md)。

## 关键限制 / Key limitations

- 合成数据只证明训练、保存、加载和服务链路可工作，不代表真实市场表现。
- 免费模式不提供全球全量实时可售票价；配置严格报价源后只显示通过该来源二次验证的有界报价，模型估价和价格曲线始终另行标注。
- 普通 Google Flights 免费查询和公共学生计划页面都不能证明当前行程已经应用学生专属价；实际学生折扣显示 `unknown`，排序不加分。公开计划链接也不等于报价级折扣证据。
- 行李和退改通常依赖具体 fare brand。没有报价级证据时保持 `unknown`。
- FAA NAS Status 只覆盖美国且只表示其列出的当前事件；“没有列出事件”不保证机场所有航班完全正常。AirLabs 免费结果可能受时间窗口、配额和最多 50 行样本限制。
- AirLabs `schedules` 是近实时且窗口有限；`routes` 是周期时刻表投影，不等于航空公司确认所选日期一定执行。两者只进入参考区，绝不填充严格 offer。
- 比较接口内部的当地正午或安全同日 +30 分钟时刻只是日期级模型/上下文参考，并由 `departure_time_basis` 区分，不是实际航班起飞钟点。
- ADSB.lol 只提供飞机位置/密度代理，不能用于声称真实延误率、取消率、机场容量或官方地面管制状态。
- NOAA、Open-Meteo、AirLabs、ADSB.lol、GDELT 和 OurAirports 的覆盖、频率、许可与可用性可能变化。
- 不应将输出用于自动购票、拒绝退款、差别定价或无人复核的重大财务/安全决策。

## License

代码以 [MIT License](LICENSE) 发布。第三方数据不随 MIT 许可证重新授权；使用者必须遵守各来源的当前条款与署名要求。
