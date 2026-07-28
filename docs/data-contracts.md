# 数据与 API 契约

本文定义 Flight Forecast Lab 的稳定边界。原始 BTS/NOAA 字段应先通过适配器转换到这里的语义，再进入训练或推理。运行中的 FastAPI `/docs` 是具体 JSON 模式的最终依据。

## 通用约定

- 字段名使用 `snake_case`；CSV 使用 UTF-8，推荐处理后数据使用 Parquet。
- 日期使用 ISO 8601 `YYYY-MM-DD`；单项预测接口时间戳必须含 UTC 偏移，例如 `2026-08-15T13:30:00-04:00` 或 `2026-08-15T17:30:00Z`。比较接口只要求 `departure_date`，不要求用户猜测航班钟点。
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
| `departure_date` | date | 是 | 不得早于出发机场当地今天、且不超过当地今天后 370 天，ISO 8601 `YYYY-MM-DD`；未来日期使用当地正午；同日查询使用仍满足 30 分钟安全余量的正午，否则使用生成时刻后 30 分钟的同日参考；若安全参考跨入次日则返回 422；这些都不代表真实航班起飞时刻 |

### 响应结构

- `context.weather`、`context.operations`、`context.news`：自动获取的三个上下文信号，均含 `[0, 1]` 数值、来源、获取时间、双语摘要和明确状态；新闻还可含最多五篇去重后的来源文章。`context.operations` 顶层是实际进入目标起飞时刻模型的信号，不能与嵌套的查询时刻 `current_snapshot` 混为一谈。票价模型不使用天气。`weather_feature_status=used` 表示准点率采用适用于目标时刻的 `live`/`forecast` 天气模型；`ignored` 表示天气为 `proxy`、`historical`、`neutral` 或 `unavailable`，准点率已切换到无天气模型。日期级比较只在出发机场参考时刻解析一次天气；每个 offer 只有在真实起飞时刻与该参考相差不超过两小时时才使用该天气，否则 `offers[].weather_feature_status=ignored` 并改用无天气模型。天气卡仍可展示回退值作参考，但必须明确提示本次准点预测已忽略天气变量。
- `offers`：严格模式下，只接收已配置且获准参与的严格来源经过初始搜索和购票选项二次验证的结果。`auto` 会查询所有这类来源，而不是在第一个确认报价后停止：SerpApi 与 SearchAPI.io 可参与；Ignav 默认保持 `ignav_quarantine`，只有 `IGNAV_STRICT_RELEASE=1` 与 `IGNAV_FREE_ACCOUNT_ATTESTED=1` 同时成立并完成受控验证后，才以独立的 `ignav_verified_fares` 身份参与。每项必须是完整连续的一至八段行程，二次响应中的已选航段与原候选完全一致，并包含真实航班号、完整当地/UTC 时刻、provider 确认的单一舱位，以及大于零的一位成人单程 USD 价格。`schedule_status=priced_offer`、与来源匹配的 `schedule_source`、`cabin_status=provider_confirmed` 与 `bookability_status=booking_option_verified` 是进入主列表的必要证据；未知航司也只有在该报价完整通过验证时才可进入。SerpApi、SearchAPI 与显式发布后的 Ignav 会分别请求二次验证各来源返回的全部合格候选；原子额度账本只批准真实剩余额度能够覆盖的调用，网络工作线程数仅限制并发、不限制候选总数。完整有序航段身份与舱位相同的多个已验证销售方/来源只保留最低最终确认价。每个 offer 都有稳定 `id`，可传入 `/v1/offer-detail`。这些结果不是所有航司或全球航班的全量清单。AirLabs/AeroDataBox 时刻、Scrape.do 聚合快照、OpenSky 航迹参考、航司目录扩展、路线级提示和纯模型航班均不得进入。
- `offers[].live_fare`：严格来源结果快照，固定为 `status=booking_option_confirmed`、`environment=production`、`currency=USD`、`price_basis=one_way_per_adult`、`traveler_count=1`、`booking_verified=true` 与 `availability_status=booking_option_verified`；`provider_code` 为 `serpapi_google_flights`、`searchapi_google_flights` 或显式发布后的 `ignav_verified_fares`，且必须与 offer `schedule_source` 和详情腿段 `data_basis` 一致。单源响应还必须与顶层 `fare_search_metadata` provider 一致；聚合响应则必须在 `fare_search_metadata.provider_runs` 中存在同 provider 且 `status=confirmed_offers` 的逐源记录。`taxes_included=null` 表示当前免费响应不能可靠证明税费是否已包含。它还包含销售方、安全 HTTPS 打开路径、`booking_url_kind`、provider offer ID、`provider_cache_hit`、`provider_cache_age_seconds` 和 `verified_at`。`verified_at` 是 provider 结果/验证时间而不是本次 API 响应时间；缓存年龄是当前响应相对该时间的整数秒。`provider_cache_hit` 是本地缓存命中或 provider 状态/时间启发式得到的布尔值，不精确证明由哪一层缓存复用。打开路径与 `booking_url_kind` 必须忠实反映来源证据，不得把需要 POST 的动作伪装成销售方 GET 直链。它与 `estimated_price_usd`、80% 模型区间及详情页 `price_curve` 相互独立；后者都不是实时报价或真实价格历史。验证后的报价仍只是来源结果快照，不保证航空公司或销售方最终结账页的库存、规则、税费或价格。
- `rankings.direct_first`、`rankings.lowest_price`、`rankings.student_first`：每组都是本次全部 `offers[].id` 的无重复完整排列；价格排序使用确认的 `live_fare.total_amount`，不使用模型估价冒充实时价格；严格结果为空时三组均为空。
- `availability_mode=strict_bookable_only`；有已确认报价时 `result_status=verified_offers_found`，否则根据原因返回 `no_verified_offer`、`fare_provider_not_configured`、`fare_provider_test_rejected`、`fare_provider_authentication_failed`、`fare_provider_rate_limited`、`fare_provider_budget_not_configured`、`fare_provider_budget_exhausted`、`fare_provider_processing`、`fare_provider_error` 或 `fare_provider_unavailable`。`fare_provider_processing` 表示 SerpApi 同一 Search ID 在有界 Archive 轮询以及最多一次已预留额度的受控重提后仍为 `Processing/Queued`；`fare_provider_error` 表示 provider 终态、HTTP 或网络错误；`no_verified_offer` 只表示所有相关查询成功完成但没有通过严格验证的购票选项。聚合响应根据逐源运行结果保留失败、处理中与额度受限信息，不会因为另一来源有报价而隐藏。`strict_mode_notice` 以中英双语说明 production 报价、覆盖与模型边界。
- `result_status=fare_provider_coverage_limited`：供应商搜索返回过候选，但真实剩余免费额度不足使验证覆盖为 `quota_limited`，而已验证子集中没有合格报价。该状态不能解释为其余候选不可购买，也不能显示成完整的 `no_verified_offer`。
- `fare_search_metadata`：给出 `status`、实际 provider、`production|disabled` 环境、本次响应观测时间、已搜索舱位、本地严格缓存命中和安全额度使用情况。只有一个来源实际运行时保留该来源 provider；两个或更多来源实际运行时，顶层必须为 `provider_code=strict_fare_aggregate` / `provider_name=Strict Fare Provider Aggregate`，并在最多四项、不可嵌套且 provider 唯一的 `provider_runs[]` 中保存每个来源的完整 metadata。聚合 offer 的实际 provider 必须对应一个 `status=confirmed_offers` 的逐源记录。`search_call_count` 统计初始搜索，兼容字段 `pricing_call_count` 统计实际发出的候选二次验证，`call_count` 为两者总和；聚合顶层为逐源计数之和。`coverage_scope=provider_returned_booking_verification_candidates` 只限定为所运行 provider 搜索实际返回且通过初步字段检查的候选，绝不表示市场全量。`eligible_candidate_count` 是合格候选数；`verification_attempted_count` 是已尝试验证数；`verified_candidate_count`、`strictly_rejected_candidate_count` 与 `provider_failed_candidate_count` 划分已尝试候选的验证结果；`search_failed_cabin_count` 单独统计未完成的舱位级搜索，不能伪装成候选。某个舱位搜索失败时，其他舱位已经独立通过二次验证的报价可以保留，但覆盖必须标为供应商不完整。`quota_skipped_candidate_count` 只统计因真实免费额度不足而未尝试的数量；`deduplicated_verified_count` 是二次验证后因完整航段与舱位相同、只保留跨销售方/来源最低确认价而移除的数量。`coverage_status=complete|quota_limited|provider_incomplete|quota_and_provider_incomplete|not_evaluated` 区分完整处理、真实额度截断、供应商响应不完整及未评估；`retry_quota_limited=true` 表示首次请求已发出但受控重试无法再次预留额度，不得把该候选误记为从未尝试。聚合覆盖同时纳入每个 `provider_runs` 的失败、处理中和额度受限状态；跨来源额度类型不一致时 `quota_limit=provider_specific`。当 `cache_hit=true` 时，三个调用计数表示本次新增调用并因此为零，候选覆盖计数保留原查询快照；双语 `notice` 明确说明这一差异。`archive_poll_count` 只统计同一 Search ID 的免费 Archive 状态读取，不加入 `call_count` 或额度预留。`diagnostics` 最多返回 10 条脱敏记录，只含观测时间、阶段、HTTP 状态、固定异常类型和格式校验后的 Search ID；不含 key、token、参数、完整 URL 或原始错误文本。本地 SQLite 最多跨重启保留 500 条相同安全字段。SerpApi 的兼容 `monthly_*` 字段按 `plan_renewal_date` 结算周期解释；SearchAPI 的 100 次硬墙是安装/账户生命周期额度，绝不称为月额度；Ignav 的最多 1,000 次也是生命周期额度。每个实际运行来源都消耗其自己的免费额度；达到真实额度上限时可返回已经验证的部分 `offers` 并明确标记截断。单个报价的 provider 缓存信息见 `live_fare.provider_cache_*`，不能用 metadata 的本地缓存命中代替。
- 候选覆盖计数中的“验证结果”只划分 `verification_attempted_count` 对应的已尝试候选；因真实剩余免费额度不足而未尝试的候选全部计入 `quota_skipped_candidate_count`，并使 `coverage_status` 为 `quota_limited` 或 `quota_and_provider_incomplete`，`quota_limit` 标明实际额度类型。因此带已验证报价的响应也可能是 partial，绝不表示全球全量航班。
- 应用没有“每次只验证 6 个”的候选上限；`MAX_BOOKING_WORKERS=6` 只控制并发网络压力。真实额度只能覆盖候选前缀时，系统先在各舱位间轮转验证直飞，再验证转机；UI 会把具体的 `hourly|monthly|lifetime|provider_specific` 限制与尝试/跳过计数显示出来。严格候选当前只支持连续的一至八段。Google Flights 的 `gl` 市场由出发机场国家决定（YYZ 为 `ca`、英国机场为 `uk`、未知国家安全回退 `us`），并在初始搜索和购票验证中保持一致。SerpApi 四舱搜索固定启用 `show_hidden=true`、`deep_search=true` 与 `async=true`，深度搜索使用独立 45 秒传输上限；若仍在处理，只按 0.5、1、2、4、8 秒有界轮询同一 Search ID。订票展开仍使用 15 秒。
- `timetable_references[]`：AirLabs `schedules`/`routes` 或 AeroDataBox 日期级 schedule 返回且通过时刻字段校验的参考行。每项只含航司、航班号、完整时刻、来源时间、`bookability_status=unverified` 和双语排除原因；没有 offer ID、舱位、价格或详情入口，也不参与排名。即使参考 schedule 与所选日期匹配，也不能替代严格报价源的购票选项验证。
- `departure_date`：用户选择的日期；`departure_time` / `departure_timezone`：服务生成的出发机场当地带偏移参考时刻和 IANA 时区。`departure_time_basis=origin_local_noon_model_reference` 表示当地正午参考；`origin_local_remaining_day_model_reference` 表示同日生成时刻后 30 分钟参考。两者仅用于模型、天气与新闻上下文，不是真实航班计划。`schedule_sample_limit=50` 是 AirLabs 参考查询行数上限；`schedule_sample_truncated=true` 只表示至少一个 AirLabs 端点报告仍有更多参考行或已返回到上限，不代表严格报价覆盖是否完整。`false` 也只表示实际查询到的 AirLabs 端点未观察到截断信号。`warnings` 为包含这些限制的中英文说明；`model_version` 为本次比较所用模型版本。

上下文状态只允许 `live`、`forecast`、`proxy`、`historical`、`neutral` 或 `unavailable`。严格主列表中的 `routing_status=provider_direct` 表示一段且 `stops=0`；`provider_itinerary` 表示二至八段且 `stops=1..7`。`segments[]` 必须按顺序连续，下一段起点等于上一段终点且下一段起飞晚于上一段抵达。旧的 `model_one_stop` 与 `model_route_unresolved` 值仍保留在 schema 供兼容读取，但严格模式不再生成对应 offer。三组排序只引用严格 offers；全球航司目录和 `route_airlines` 不再自动生成候选航班。

每个 offer 的航班计划证据使用以下字段：

| 字段 | 语义 |
| --- | --- |
| `schedule_status` | 主列表固定为 `priced_offer`；`live_schedule`、`recurring_timetable_projection` 与 `model_scenario` 仅为参考/兼容语义 |
| `schedule_source` | 主列表必须与报价来源一致：`serpapi_google_flights_booking`、`searchapi_google_flights_booking` 或仅供已显式发布 `ignav_verified_fares` 使用的 `ignav_verified_booking`；参考行使用 `airlabs_schedules`、`airlabs_routes` 或 `aerodatabox_schedule` |
| `routing_status` / `stops` | `provider_direct/0` 或 `provider_itinerary/1..3`；模型内部参考不得公开声称为真实航段 |
| `flight_number` | 摘要使用第一航段真实航班号；完整行程见最多八项 `segments[]` |
| `scheduled_departure_local` / `scheduled_arrival_local` | 第一段出发与最后一段抵达的当地钟点，必须通过机场 IANA 时区、日期和先后顺序校验 |
| `scheduled_departure_utc` / `scheduled_arrival_utc` | 对应第一段出发与最后一段抵达的 UTC 绝对时刻 |
| `provider_flight_status` / `schedule_observed_at` | 主列表状态为 `booking_option_verified`，来源时间是 `booking_token` 购票选项验证时间；AirLabs 参考的 observed time 仍按 schedules 抓取时间或 routes record `updated` 解释 |

严格报价层会运行所有已配置且获准参与的来源，并为每个来源的四种请求舱位解析带该来源二次验证令牌/标识且通过初步字段检查的候选；SerpApi/SearchAPI 支持时可使用 `deep_search=true`、`show_hidden=true` 扩大可见范围，但仍不保证全量覆盖。SerpApi、SearchAPI 与显式发布后的 Ignav 会请求验证各自搜索实际返回的全部合格候选；原子账本仅在对应账户仍有真实额度时批准调用，额度不足而未验证的部分必须计入 `quota_skipped_candidate_count` 并标记 `partial` / `quota_limited`。只有二次返回的已选航段与原候选逐段一致，并仍满足精确 O&D、出发日期、未来起飞、连续一至八段、每段真实航班号和完整时刻、全程同一请求舱位、一位成人单程 USD 正数价格，且至少一个购票选项具有销售方、匹配航班号和安全的 HTTPS 打开路径，结果才可进入主列表。同一完整航段序列和舱位通过多个销售方或来源验证时按最终确认价格只保留最低者，且保留获胜来源的证据。打开路径必须忠实保留 provider 的 GET/POST 语义，不伪造普通直链或“已选行程”。单个来源的 `no_results` 不会阻止其他已配置来源执行；只有所有完成来源均没有经自身二次验证的报价时才可返回完整最终空结果。

SerpApi Free 目前包含每个 `plan_renewal_date` 结算周期 250 次成功搜索及每小时 50 次请求；初始搜索和 `booking_token` 查询共用额度。候选数量随航线、日期和 provider 响应变化，因此单次比较的验证调用数也会变化。服务使用持久化单一计数和 `SERPAPI_MONTHLY_LIMIT` 在本地 hard limit 前保守预留所有尝试；字段名保留 monthly 兼容性，但周期不是自然月。`Processing/Queued` 只轮询固定 Archive 地址和白名单格式 Search ID；有界轮询仍未完成、provider 搜索错误、传输错误或 HTTP 408/425/5xx 才允许在再次原子预留额度后重提一次，第二次绝不触发第三次。HTTP 400、认证失败、429 与严格解析拒绝不重试。即使 provider 缓存请求按官方规则免费，本地也预留，所以 `monthly_calls_used` 可能高于 provider 计费用量并提前停止。省略上限时默认 250，大于 250 被钳制为 250，非法或非正值也不会解除上限。额度不足时，已经成功验证的报价可保留，剩余候选以 `quota_skipped_candidate_count`、`retry_quota_limited`、`coverage_status` 和 `quota_limit` 明确标记截断；认证失败、HTTP 限流、provider 不可用、无结果或字段验证失败仍返回带 `fare_search_metadata` 的结构化结果，不得启用 AirLabs 或模型补位。免费额度、覆盖、字段和条款可能变化；缺少某航司、舱位、航线、销售方或日期时必须如实为空，不能伪造航班号、精确时刻、舱位或价格来“补全”列表。

SearchAPI 的本地硬墙是安装/账户生命周期一次性 100 次请求，不能命名或解释为月额度。Ignav 默认 `ignav_quarantine`，即使配置了 key 也不能产生严格报价；只有 `IGNAV_STRICT_RELEASE=1` 与 `IGNAV_FREE_ACCOUNT_ATTESTED=1` 同时成立并完成受控验证后，才使用独立的 `ignav_verified_fares` 身份及最多 1,000 次生命周期额度。任何 `ignav_quarantine` 结果都不得进入 `offers`。

AirLabs `schedules` 与 `routes` 在严格比较中仅用于 `timetable_references` 和机场运行上下文。取消、已起飞、active、landed、departed 或其他明确非未来状态的 live identity 会先覆盖相同 routes 投影，禁止周期投影将其“复活”。两个端点仍需要可选的 `AIRLABS_API_KEY`，并受免费层时间窗口、配额、每次最多 50 行和全球覆盖限制；所有调用点与进程在网络请求前共用一个原子 SQLite 月度账本。若设置 key 但没有明确有效的 `AIRLABS_MONTHLY_CALL_LIMIT`，调用 fail closed；响应 `request.key.limits_by_month` 与 `limits_total` 只保存脱敏数字，并且只能收紧后续硬墙。任一端点的 `request.has_more` 为真或返回行数达到 50 时，响应将 `schedule_sample_truncated` 标为真。本演示为控制免费配额和调用量不继续分页。AirLabs 不提供座位库存或实时报价，因此其任何行都保持 `bookability_status=unverified`。

Scrape.do 固定为参考源：每月最多 1,000 credits，每个未缓存参考调用预留 10 credits，不能产生 offer、购票链接或详情页。OpenSky 无凭据匿名模式只提供当前航迹密度参考，不能证明未来航班、票价、库存或延误。AeroDataBox 只提供日期级时刻参考；只有可信 RapidAPI 免费计划限额/剩余/重置头可以建立或更新账期，没有可信重置证据时执行安装生命周期 600 API 单位硬墙，不能因自然月变化自行恢复，也不能进入严格列表。

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

严格模式生成的一段 offer 为 `provider_direct`、`stops=0` 和 `punctuality_basis=direct_leg_model`；二至八段 offer 为 `provider_itinerary`、`stops=1..7` 和 `punctuality_basis=multi_leg_independence_model`。`model_one_stop`、`model_route_unresolved`、`two_leg_independence_scenario` 与 `route_only_model` 仅作为旧响应兼容枚举保留，当前严格比较不生成这些模型航班。

政策状态采用严格证据语义。`unknown` 不等于“不包含”；当前免费严格报价来源不能验证当前行程是否获得实际学生专属折扣，因此 UI 显示 `unknown`，排序不加分。公开学生计划只标为 `program_available`，也不能当成当前航线、日期和舱位已经获得实际学生折扣。只有独立的报价级证据才可使用 `confirmed_free`、`confirmed_included` 或 `confirmed_discount`；当前严格链没有这类学生折扣证据。

学生优先排序依次比较：最低 `live_fare.total_amount`、免费托运行李、已确认实际学生折扣、免费改签/退票、年龄与身份验证门槛。`program_available` 本身不满足“实际折扣”条件，只在最后一级使用其公开年龄和验证信息。没有独立证据的条件保持 `unknown`，评分为中性，不因缺失而加分。模型估价与价格曲线不代替 live fare 参与最低价格判断；后续条件主要用于同价时打破平局。

## Offer 详情 API

页面：`GET /details/offer`；数据端点：`POST /v1/offer-detail`。

主页中的每个 offer 都提供二级详情入口。请求字段：

| 字段 | 类型 | 必填 | 语义 |
| --- | --- | --- | --- |
| `origin` | string | 是 | 与比较请求相同的三位机场代码 |
| `destination` | string | 是 | 与比较请求相同的三位机场代码 |
| `departure_date` | date | 是 | 与比较请求相同的出发日期 |
| `offer_id` | string | 是 | 比较响应中 `offers[].id` 的不透明稳定标识；客户端不得自行拼接或修改 |
| `force_refresh` | boolean | 否 | 默认 `false`，初次详情加载复用 5 分钟严格缓存；只有用户点击“刷新并重新查询”时传 `true`，重新执行四舱搜索并在免费额度与供应商响应范围内验证实际返回的候选 |

响应中的 `offer` 沿用比较响应的 live fare、模型估价、准点、政策、舱位和 schedule 字段。详情初载 `force_refresh=false` 优先复用 5 分钟严格缓存；`force_refresh=true` 才重新执行 Google Flights 搜索和 `booking_token` 购票选项验证。两种路径都只接受仍存在于当前严格结果中的 offer，已过期、旧周期/模型或无法重新确认的 offer ID 返回 404。主页比较请求最长等待 600 秒，并保留超时或失败信息；每个来源都会在其真实剩余额度内请求验证全部合格候选。`itinerary` 提供：

- `kind=direct|one_stop|multi_stop`、`time_basis=provider_schedule`、总距离与总时长；兼容 schema 可保留 `route_unresolved|model_duration_only`，但严格详情不生成它们；
- `legs[]` 为一至八项购票选项已验证航段，每段包含起终点、营销/承运航司、航班号、当地/UTC 时刻、距离、时长、确认舱位，以及 provider 返回时可用的 booking class、fare basis/brand、托运行李和航站楼/机型字段；`data_basis` 必须与 `live_fare.provider_code` 对应：SerpApi 为 `serpapi_booking_confirmed`、SearchAPI 为 `searchapi_booking_confirmed`、显式发布后的 Ignav 为 `ignav_verified_booking_confirmed`；
- `layovers[]` 最多三项，列出中转机场和由相邻已确认时刻计算的停留分钟数；`layover_status=provider_confirmed`。直飞没有中转项；
- `schedule_status` / `schedule_source` / `schedule_observed_at`、`fare_search_metadata`、`weather_feature_status`、双语 `fallback_reason` 和 `notice` 明确说明报价时间、调用状态、天气是否进入准点模型及证据边界；AirLabs 的 `schedule_sample_*` 只描述旁侧参考样本，不改变该报价的确认状态。
- `price_curve`：`status=model_projection`、`basis=verified_fare_anchored_synthetic_trajectory`、`calibration_method=log1p_offset_to_verified_fare`、`currency=USD`、已验证报价锚点及来源、原始模型起点、校准偏移、起止日期、生成时间、是否超出主要 180 天训练提前期，以及最多 371 个逐日点。每点含 `quote_date`、带时区 `quote_time`、距离起飞天数、锚定后的模型估价和经同一变换得到的 80% 区间；第一点必须精确等于当前 `live_fare`。曲线仍是改变模拟查询日期得到的演示模型轨迹，不是未来真实或可购买报价。
- `historical_market_context`：可选的独立历史市场证据，来自严格报价响应随附的 `price_insights.price_history`，不新增供应商调用。它必须与当前 `origin`、`destination`、`departure_date` 和 `live_fare.cabin_summary` 全部一致，并保留提供商、提供商观测时间及美元价格点；语义仅为同航线、同计划出发日期、同舱位市场历史，不是具体航班、销售方或购票选项的价格历史。字段缺失或任一点格式异常时整段丢弃，不影响严格报价，也不用模型补造。

`serpapi_booking_confirmed`、`searchapi_booking_confirmed` 与 `ignav_verified_booking_confirmed` 只能用于各自来源经过初始搜索和二次购票选项验证的航段；Ignav 的值仅允许配合已显式发布的 `ignav_verified_fares`，不能用于 `ignav_quarantine`。AirLabs 的 `airlabs_live_schedule`/`airlabs_recurring_timetable_projection`、AeroDataBox 时刻、OpenSky 航迹和 Scrape.do 聚合数据只能出现在参考语义中。无可用严格凭据、无覆盖、本地额度耗尽、认证失败、限流、超时、空结果或字段不完整都返回结构化结果，绝不补造航班号、精确钟点、舱位、报价或模型航班；所有配置并获准的严格来源都会执行，其逐源证据通过 `provider_runs` 保留。保留 offer 的舱位为 `provider_confirmed`，购买状态为时点性的 `booking_option_verified`；HTTPS 打开路径由 provider 证据产生，并用 `booking_url_kind` 标明是否为直接 GET 跳转，但这不等于出票或最终结账价格保证。聚合严格覆盖也只是这些免费来源在各自额度和响应范围内的并集，不等于全球全部可购库存。

## 上下文详情 API

两个上下文详情接口都要求 `origin`、`destination`，并且必须且只能提供一个 `departure_date` 或兼容字段 `departure_time`。主页链接传递 canonical `departure_date` 与当前 `departure_time_basis`；详情页每次刷新只向 API 提交日期，由服务根据新的 `generated_at` 重新生成安全参考。因此旧的同日 +30 分钟参考过期后刷新不会因重复提交旧钟点而 422。响应返回重新计算后的 `departure_time` 和 `departure_time_basis`：`origin_local_noon_model_reference` 是内部正午参考，`origin_local_remaining_day_model_reference` 是同日生成时刻后 30 分钟参考；两者都不是真实航班钟点。兼容调用可只提交 `departure_time`；无偏移时间按出发机场当地时区解释，必须在未来且不超过 370 天，响应 basis 为 `legacy_input`。页面路由 `GET /details/weather` 与 `GET /details/news` 不直接返回数据，而是读取查询参数并调用下列 API。

### `POST /v1/context/weather-detail`

响应包含航线、带时区的模型/天气参考时刻、`departure_time_basis`、根据航线模型时长估算的抵达参考、生成时间，以及 `origin_weather` 和 `destination_weather`。日期级参考和抵达参考都不是航班计划。每个机场对象包含：

- 机场代码/名称、可选 ICAO、IANA 时区和目标时刻；
- `current` 与 `target` 天气：温度、WMO 天气代码及双语描述、风速、阵风、降水、降水概率、能见度、风险值；
- 目标时刻前后 12 小时最多 25 个 `hourly` 点；
- 最多五项 `risk_components`：天气代码、风、阵风、降水、能见度；
- 当前实现最多返回一条最近有效的 `METAR` 和一条适用于目标时刻的 `TAF`；每条含原始报文、签发/有效时间、风险和保守的双语解释（响应模式为未来扩展保留最多四条容量）；
- `aviation_metadata`：明确区分 NOAA 报告可用、无适用报告、缺少 ICAO、部分产品失败或服务不可用，并给出来源、时间和双语原因；
- `metadata`：总体天气风险的状态、实际主导来源、观测时间、有效期及可选双语回退原因。当适用的 NOAA 报告风险更高时，总体风险和这些时间字段都以该报告为准。

到达时刻是模型估计，不是实际航班计划；自动 METAR/TAF 解读不能替代官方航空气象简报。页面每 10 分钟自动刷新，并提供手动刷新。

### `POST /v1/context/news-detail`

响应包含航线、模型/新闻参考时刻及其 `departure_time_basis`、生成时间、最多 20 篇文章、`route_raw_risk`、`departure_attenuation_factor`、`model_effect`、`model_signal`、元数据、双语摘要与 GDELT 索引时间提示。参考时刻不是航班计划。`route_raw_risk` 是详情页较大文章集的解释性分数；`model_effect` 与 `model_signal.value` 是主预测上下文实际使用的新闻输入，`model_signal` 还给出其状态、来源、观测时间和双语摘要。每篇文章包含原始 `title`、清理后的 HTTP(S) `url`、来源域名、可选语言、`indexed_at`、风险 `category`、最多 12 个 `matched_risk_terms`、`raw_score`、`recency_factor` 和 `weighted_score`。

允许的类别为 `airport_closure`、`airspace_conflict`、`labor_strike`、`extreme_weather`、`cancellation_delay`、`security_cyber` 与 `other_disruption`。文章时间是 GDELT 观察/索引时间，不保证是媒体发布时间；标题保留来源语言。页面每 15 分钟自动刷新，并提供手动刷新。

## 服务辅助端点

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 进程与模型加载健康状态；不代表模型仍然准确 |
| `GET /v1/model-info` | 模型版本、训练来源、时间和可用任务信息 |
| `POST /v1/compare` | 用三个输入处理四舱搜索实际返回的全部合格候选；SerpApi、SearchAPI 与显式发布后的 Ignav 仅受各自真实额度和供应商响应限制，返回已严格验证 offer、partial / quota-limited 覆盖信息及三类排序；不表示全球全量 |
| `GET /details/offer` | 每个比较 offer 的中英双语航班/模型行程详情页面 |
| `POST /v1/offer-detail` | 按 `offer_id` 返回再次通过购票选项验证的 provider schedule；无法验证时返回结构化失败 |
| `GET /details/weather` | 中英双语天气详情页面；查询参数由主页生成 |
| `GET /details/news` | 中英双语新闻详情页面；查询参数由主页生成 |
| `POST /v1/context/weather-detail` | 出发与到达机场的当前/目标天气、趋势和 METAR/TAF 详情 |
| `POST /v1/context/news-detail` | 最近七日最多 20 篇中断新闻及文章级风险解释 |
| `POST /v1/destination/places` | 按目的地机场返回脱敏的 OSM 景点或酒店列表及真实查询覆盖半径 |
| `POST /v1/destination/place-detail` | 返回 OSM 地点详情、道路图路线，以及 Transitous 或配额受控 SerpApi Google Maps Directions 回退返回的完整公共交通行程；两者均无真实结果时返回明确不可用状态 |
| `POST /v1/destination/hotel-prices` | 用户显式触发 Google Hotels 住宿报价搜索；与严格航班共用 SerpApi 额度 |
| `POST /v1/destination/hotel-price-detail` | 用 `hotel_id` 或 OSM `place_id` 二选一严格确认同一家酒店，并返回真实房型价格、跨平台评分/评价与机场交通 |

健康响应不得暴露本地绝对路径、密钥或原始训练记录。所有 provider 凭据都不得进入浏览器/前端、仓库或应用日志；服务端只把凭据发送给对应的 HTTPS provider，完整外部请求 URL 不得记录。模型信息应能让调用方识别是否误用了 `synthetic` 演示模型。

## 派生特征规则

推荐从原始请求一致地派生以下信息，而不是要求调用方重复提供：

- `lead_time_hours = departure_time - quote_time`；
- 起飞月份、星期、小时和周末标志；
- 由出发机场 IANA 时区得到的本地月份、星期和小时；训练表可显式提供 `departure_local_month`、`departure_local_weekday`、`departure_local_hour`，服务推理时会自动生成；
- 标准化航线键（方向性是否保留需由任务声明）；
- 根据机场坐标或版本化航线表推算距离/时长，并保留来源标记；
- 在预测时刻获取天气预报、机场运行和近七日新闻；新闻优先使用 15 分钟航线缓存，失败时只允许使用不超过 6 小时且标为 `historical` 的旧缓存，否则中性回退；
- `news_disruption_index` 同时进入票价与准点模型；票价模型不使用天气；机场运行信号进入准点模型，天气仅在状态为 `live` 或 `forecast` 时进入含天气准点模型，其他状态通过 `weather_feature_status=ignored` 切换到无天气模型；
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

当前演示模型使用 `schema_version = 3`；该版本保留服务端自动解析天气与机场运行的接口，并增加无天气准点模型与 `weather_feature_status`，确保不适用的 proxy/历史天气不会被静默当作准点预测输入。

API 客户端应以 `/v1/` 为稳定主版本边界，并读取 `/v1/model-info` 判断模型来源与版本。批量推理时还应保存请求模式版本、模型版本和预测时间，以便复现。
