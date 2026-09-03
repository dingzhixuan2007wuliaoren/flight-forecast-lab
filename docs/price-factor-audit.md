# 价格预测因素审计

审计日期：2026-09-02

本文记录 Flight Forecast Lab 对机票与酒店价格因素的取舍。目标不是把所有看似相关的变量都塞进模型，而是只使用在预测时确实可获得、训练与推理语义一致、来源可追溯且不会把标签偷渡进特征的信号。

## 1. 判定标准

一个因素只有同时满足以下条件，才可以标为“采用”：

1. **服务时可得**：在生成预测的时刻已经存在，不能等到起飞、入住或供应商结算后才知道。
2. **可作 as-of 连接**：历史训练行只能看到当时已发布的版本，而不是今天回填到过去的修订值。
3. **训练与推理同义**：单位、时间口径、市场、产品范围和缺失规则一致。
4. **不等价于目标**：同一次搜索的实际价格、由该价格直接计算的排名或“便宜/昂贵”标签不能进入模型。
5. **覆盖可解释**：来源缺失时有 `unknown`、缺失指示或独立的精简模型；不能把“查不到”伪装成零。
6. **经时间外评估**：在固定时间切分上优于更简单的基线，并且没有以更差的区间覆盖或明显的分组退化换取表面上的平均误差下降。

这里的“采用”分两种：

- **当前已实现**：已经进入当前特征构造或本次模型契约；
- **有条件采用**：经济上合理，但只有在真实、合法、带可用时间戳的数据完成接入和消融后才可进入生产模型。

合成数据上的改善只证明代码能学习预先写入的数据关系，不能证明真实世界的预测精度。

## 2. 结论摘要

| 任务 | 当前采用 | 有条件暂缓 | 明确拒绝 |
| --- | --- | --- | --- |
| 机票 | 航线/航司/舱位、经停、距离/时长、提前期和出发日历；`route_scope` 与 `carrier_service_model` 已进入受门槛约束的候选消融，只有选择段证据通过才启用；新闻仅按严格的预测时点快照语义使用 | 最新已发布的滞后运力/竞争、滞后燃油成本、官方节假日/重大活动、严格早于预测时刻的自有价格快照、明确的票价产品条款 | 同次搜索的实际价与 `price_insights`、目标航班最终载客率/剩余座位、事后新闻/实际延误、未来实际天气、把 O&D 乘客数或同周期均价当行级特征 |
| 酒店 | 目的地、物业类型、入住提前期、住宿晚数、成人数、入住日历与周末晚占比；增强模型再使用星级、评分、评论量、位置距离、设施数及免费取消状态 | 官方节假日/活动、严格滞后的城市市场价格、可验证的历史供需/库存、房型与餐食条款、品牌/连锁、覆盖完整的周边竞争 | 当前真实房价/总价/供应商胜出来源、由价格产生的优惠标签、酒店令牌或预订链接、今天的评分回填历史、当前天气代表未来入住天气、酒店名称/ID 记忆型捷径 |

## 3. 机票价格因素

### 3.1 当前基础因素与已实现候选

| 因素 | 特征 | 预测时可得性与理由 | 主要边界 |
| --- | --- | --- | --- |
| 产品与市场身份 | `origin`、`destination`、`route`、`airline`、`cabin` | 请求或经严格报价确认前已经知道；路线、承运人和舱位定义了不同产品市场 | 航司代码复用和营销/运营口径必须版本化；未知类别不得猜测 |
| 行程结构 | `stops`、`duration_minutes`、`distance_km` | 经停、计划时长和距离与产品便利性及成本相关；服务可从已确认行程或机场坐标推导 | 训练时必须使用真实或验证后的计划值；不能用实际飞行时长回填 |
| 购买提前期 | `days_until_departure` | 动态定价研究持续表明销售截止期与有限库存会改变价格；由 `departure_time - quote_time` 严格派生 | O&D 抽样售票数据通常没有准确询价时间，不能捏造 `quote_time` |
| 出发日历 | 月、星期、小时的正余弦，以及 `is_weekend` | 在预测时完全可知，表达季节、工作日和时段差异 | 必须按出发机场本地时间计算；不能先丢弃 UTC 偏移 |
| 路线范围候选 | `route_scope = domestic / international / unknown` | 由起终点机场国家代码派生，无需额外付费调用；帮助模型区分跨境和境内产品制度 | 已实现但不保证进入最终模型；机场国家来自版本化机场目录，无法解析时必须为 `unknown`，不能按机场代码外观猜测 |
| 航司服务模式候选 | `carrier_service_model = full_service / hybrid / low_cost / unknown` | 当前版本化航司目录在预测前可解析；有助于已知航司之外的产品定位泛化 | 已实现但不保证进入最终模型；服务模式会随时间和品牌变化，真实训练需有效日期映射，未知或冲突时使用 `unknown` |
| 近期中断新闻 | `news_disruption_index` | 只有报价时刻已经观察到、与航线相关且有时间戳的新闻快照才符合服务语义 | 演示训练中的关系是合成的；不得用事后报道、今天的新闻回填历史，来源失败时只能用明确的中性/历史回退 |

提前期、容量和竞争的经济依据可见航空动态定价研究：有限容量、随机需求、距离销售截止期的时间以及竞争者状态共同影响价格。参见 [Hortaçsu、Öry 与 Williams，Dynamic Price Competition](https://www.ftc.gov/system/files/ftc_gov/pdf/williamshortacsuory.pdf)、[Escobari 与 Gan，Price Dispersion under Costly Capacity and Demand Uncertainty](https://www.nber.org/papers/w13075) 以及 [Borenstein 与 Rose，Competition and Price Dispersion](https://www.nber.org/papers/w3785)。这些研究支持因素方向的合理性，不替代本项目自己的时间外验证。

`route_scope` 的国家信息可以来自项目已使用的 [OurAirports 公共领域 CSV 与数据字典](https://ourairports.com/data/)。该来源会更新且不保证完全准确，因此训练产物必须记录目录版本或哈希。

当前机票训练同时拟合四个候选：legacy、仅 `route_scope`、仅 `carrier_service_model`、两者合并。四者使用同一训练行；只在时间上更晚的 selection 段比较 MAE。为避免某一天因候选报价较多而支配选择，先在每个报价日期内求 MAE，再对日期等权平均。新增候选必须相对 legacy 至少改善 US$0.50、拥有至少五个独立报价日期块，并且通过以报价日期为整块的配对 bootstrap 下界门槛。三个非 legacy 候选使用 Bonferroni 校正，将家族置信水平控制为 95%；系统先逐一筛选，再从全部合格者中选 selection MAE 最低者，否则保留 legacy。选定后使用再往后的 conformal calibration 段计算 80% 美元绝对残差区间，最终 test 不参与任何选择；同一时间戳的提供商快照不会跨段拆分。这一消融、门槛、逐候选选择 MAE、bootstrap 证据、最终选中因素和 `synthetic_validation_only` 都写入航班产物元数据。

当前仓库内确定性演示产物的实际结果是保留 **legacy**：按报价日期等权后，enriched 在 selection 的 MAE 为 US$73.78，legacy 为 US$74.18，平均改善约 US$0.40，低于 US$0.50 实际门槛；其 Bonferroni 校正后的 98.33% 配对 bootstrap 改善区间约为 US$−1.76 至 US$2.65，下界也未大于零。仅 `route_scope` 和仅 `carrier_service_model` 同样未通过。因此两个新因素已接入、可审计，但没有被当前默认部署模型强行启用。这些数字只来自合成演示样本，不是现实航线效果估计。

### 3.2 有条件暂缓

#### A. 滞后航线运力与竞争

候选包括：

- 最新已发布期的航线座位数与计划班次数；
- 最新已发布期的航线载客率；
- 按座位或旅客份额计算的承运人数、市场份额或 HHI；
- 上述来源的 `source_period_end`、`published_at` 和 `context_age_days`。

[BTS T-100](https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=Nv4+Pn44vr45&gnoyr_VQ=GDM) 是无需密钥的官方月度来源，包含座位、旅客、计划/实际班次和载客率，适合离线构建美国承运人覆盖范围内的滞后市场上下文。

暂缓原因：

- T-100 有发布滞后且覆盖范围有限；
- 目标月最终旅客数和载客率在报价时尚不可知；
- 国际或小型市场容易大量缺失；
- 如果直接使用与标签同周期的汇总，会把未来需求混入特征。

只有在实现“取 `quote_time` 当时已经发布的最后一期”连接、保留数据年龄、提供缺失回退，并通过消融后，才能有条件采用。特征名不应伪装成固定的 `lag_1m`，除非真实发布日期确实保证一个月滞后。

#### B. 滞后燃油成本

[EIA 美国墨西哥湾沿岸航空煤油周度价格](https://www.eia.gov/dnav/pet/hist/eer_epjk_pf4_rgc_dpgW.htm) 可无密钥下载；[BTS Fuel Cost and Consumption](https://transtats.bts.gov/FUEL/) 还提供美国航空公司月度成本和消耗。

燃油成本确有传导，但幅度、时滞和航司类型存在异质性；2025 年开放研究也显示不同商业模式的传导幅度不同：[Cost pass-through in the U.S. aviation industry](https://doi.org/10.1016/j.jairtraman.2025.102823)。因此它只适合作为带发布日期、地区和数据年龄的滞后宏观特征，不能把起飞日未来燃油价格当成已知值，也不能把美国区域指标冒充全球逐航班成本。

#### C. 节假日、活动与更细的季节峰值

节假日和大型活动可能造成需求峰值。美国、加拿大和英国均有官方无密钥日历，例如 [美国 OPM 联邦节假日规则](https://www.opm.gov/policy-data-oversight/pay-leave/pay-administration/fact-sheets/holidays-work-schedules-and-pay/)、[Canada.ca 公共假日](https://www.canada.ca/en/revenue-agency/services/tax/public-holidays.html) 与 [GOV.UK Bank Holidays API](https://www.api.gov.uk/gds/bank-holidays/)。

暂缓原因是本项目覆盖全球机场，节日有国家、省州、宗教和临时公告差异；活动日期还可能取消或变更。只有版本化来源能同时覆盖训练期与服务期，并记录公告可用时间时，才可加入 `days_to_nearest_holiday`、`is_holiday_window` 或活动容量等特征。简单硬编码“十二月一定涨价”不算可靠接入。

#### D. 严格滞后的自有价格快照

同航线、日期、舱位的早先报价滚动中位数通常会很有信息，但必须满足：

- 每个价格点的 `observed_at < prediction_time`；
- 仅使用相同市场、行程范围、舱位、币种和税费口径；
- 所有滚动统计只用当前行之前的数据；
- 训练、校准、测试分别按时间顺序计算，不能先在全表聚合再切分。

SerpApi 的 [Google Flights Price Insights](https://serpapi.com/google-flights-price-insights) 同时返回 `lowest_price`、`price_level`、`typical_price_range` 与 `price_history`。这些字段可作为独立市场证据展示，但若模型正在预测同一次搜索的价格，把它们直接作为特征会产生循环和目标泄漏。项目不应为了制造训练集自动消耗有限的严格报价额度；只有依法留存、严格早于预测时点的自有快照才进入本候选。

#### E. 票价产品条款

可退改、行李额度、票价品牌、销售渠道和是否联程会影响最终产品价格。严格预订结果可能提供部分字段，但这些字段不是普通价格预测请求的稳定输入，而且获取它们时通常已经获得真实价格。只有把任务明确改成“给定完整票价产品属性估计合理价格”、历史快照也保留同一属性时，才适合训练单独的产品级模型。

### 3.3 明确拒绝

| 因素 | 拒绝原因 |
| --- | --- |
| 同一次搜索的 `live_fare`、最低价、价格等级、典型区间 | 与目标同源或由目标直接产生；可以做显示锚点，不能进入原始价格模型 |
| 目标航班最终乘客数、最终载客率、最终剩余座位 | 起飞后或售罄后才完整可知，是结果信息；只能使用当时可得的库存快照或已发布的滞后市场聚合 |
| 实际延误、取消、实际飞行时长、事后新闻 | 都发生在票价预测之后 |
| 起飞日实际天气，或把今天的天气用于数月后的航班 | 预测时不可知或不适用于目标时刻；当前票价模型继续不使用天气 |
| DB1B/DB1C `passengers` 作为单张票行级预测特征 | 它是抽样权重/汇总数量候选，不是消费者询价时可知的同粒度输入 |
| 同周期平均票价、同周期市场份额或载客率 | 汇总中包含当前标签及未来记录；必须改成可验证的上一已发布期才可能使用 |
| 航班号、酒店式高基数 ID、预订令牌 | 容易记忆样本且对新产品无泛化价值；令牌还可能敏感、短期失效 |
| 碳排放、机型、腿部空间直接全部加入 | 当前与距离、时长、舱位高度重叠，服务覆盖不一致；没有消融证据前不增加复杂度 |

## 4. 酒店价格因素

### 4.1 目标与使用场景

酒店模型目标为 `nightly_price_usd`：针对指定目的地、物业、入住/退房日期与成人数的**来源展示平均每晚美元价格**。当前演示训练数据为 deterministic synthetic demo，输出不是实时可订报价，也不保证包含税费或最终结账费用。

真实训练适配器必须把税前价和含税/费展示价分成不同目标，不得混在同一 `nightly_price_usd` 列。`total_price` 也不能作为 nightly 目标的特征；晚数只是产品条件，真实总价可能含一次性费用，不能一律假设为 `nightly × nights`。

模型分两层：

- **core**：在不为模型额外调用付费酒店来源时也能得到的住宿条件；
- **enriched**：只有调用方已经有同一时点的物业质量与位置快照时才使用。获取增强字段时往往也已经获得真实价格，因此增强模型只作为估值基准或缺价回退，不能替代真实报价。

### 4.2 当前采用：core

| 特征 | 语义与可得性 |
| --- | --- |
| `destination` | 目的地市场；来自请求或已解析机场服务城市 |
| `property_type` | 只接受 `hotel`、`hostel`、`guest_house`、`motel`、`apartment` 五个规范化类型；当前训练与 API 对其他/未知值 fail closed，不把它们映射为 `unknown` |
| `adults` | 本次住宿成人数，属于查询条件 |
| `days_until_check_in` | `check_in_date - quote_time`，必须为正且按服务接收时刻计算 |
| `stay_nights` | `check_out_date - check_in_date`，明确表达产品长度 |
| `weekend_night_share` | 住宿各晚中周末晚的比例，比仅看入住日是否周末更准确 |
| `check_in_month_sin/cos` | 入住月份的循环编码 |
| `check_in_weekday_sin/cos` | 入住星期的循环编码 |

酒店提前预订、季节、活动、竞争、客房品质与服务会共同影响在线房价；不同城市与酒店档次的提前预订轨迹不一定单调。参见 [Guizzardi、Pons 与 Ranieri，Advance booking and hotel price variability online](https://cris.unibo.it/handle/11585/629730) 和 [Lee，Modeling and forecasting hotel room demand based on advance booking information](https://doi.org/10.1016/j.tourman.2017.11.004)。因此模型应学习非线性关系，而不是硬编码“越晚订一定越贵”。

### 4.3 当前采用：enriched 候选

增强模型在 core 基础上加入：

| 特征 | 语义与处理 |
| --- | --- |
| `hotel_class` | 1–5 星级；缺失不能默认为零星 |
| `rating` | 当时已经显示的规范化评分；需保存评分量表和观察时间 |
| `log_review_count` | 由当时的 `review_count` 使用 `log1p` 派生，降低长尾影响 |
| `distance_from_city_center_km` | 物业坐标到版本化目的地中心的球面距离 |
| `distance_from_airport_km` | 物业坐标到目的地机场的球面距离 |
| `amenity_count` | 规范化、去重后的设施数量；只表示来源返回的可见设施，不表示完整设施清单 |
| `free_cancellation` | 与该价格证据同一不可拆分行返回的免费取消状态 |
| `free_cancellation_known` | 区分明确的 `false` 与来源未提供；禁止把未知当作不可取消 |

研究支持星级、消费者评分和位置属性的重要性：[Determinants of online hotel room prices](https://doi.org/10.1108/IJCHM-09-2018-0707) 发现星级和顾客评分具有解释力；[Why are hotel room prices different?](https://orca.cardiff.ac.uk/id/eprint/116727/) 发现酒店品质属性以及距机场、道路与景点的位置属性存在空间异质关系。

项目已接入的 [SerpApi Google Hotels](https://serpapi.com/google-hotels-api) 可返回入住日期、成人数、坐标、星级、评分、评论量、设施、免费取消、每晚价和总价；[SearchAPI.io Google Hotels](https://www.searchapi.io/docs/google-hotels-api) 还可能返回位置、公共交通和机场可达性评分。两者都需要凭据和有限额度，因此：

- 不得为了给模型补特征而额外发起酒店查询；
- 不得把凭据、完整外部 URL 或短期房源令牌写入训练数据；
- 训练必须保存各属性自己的 `observed_at`，不能用今天的评分与设施回填旧价格；
- 只有增强模型在相同时间选择段的 MAE 至少比 core 低预注册的 US$0.50，才选择增强模型，否则部署 core。

物业坐标还可来自现有 OSM/OurAirports 上下文，但公共 [Nominatim 使用政策](https://operations.osmfoundation.org/policies/nominatim/) 明确限制重度使用并要求低频、有效 User-Agent 和署名。不得用公共 Nominatim 批量制造训练集；应复用请求中已取得的坐标、缓存结果或离线数据。

酒店模型与航班产物分开保存在 `hotel_model_bundle.joblib`（`hotel_model_schema_version = 1`）。`train-demo` 默认同时训练确定性合成酒店数据；`train-csv --hotel-price-csv ...` 可训练独立自定义酒店产物；`predict-hotel-price` CLI 与 `POST /v1/predict/hotel-price` 只读取本地产物，不调用外部酒店 provider，也不消耗额度。HTTP API 对 core/enriched 的可选字段如实记录已提供与缺失集合，并明确返回 `data_mode` / `synthetic_demo`。

当前仓库内确定性酒店演示产物选择了 **enriched**：selection MAE 为 US$31.46，而 core 为 US$71.21；未参与选择的最终合成 test 上，enriched MAE 为 US$33.46、训练中位数基线 MAE 为 US$126.31、80% 区间经验覆盖率约 79.78%。这只能验证实现能够学习生成器中人为注入的因素关系，不能证明现实酒店价格精度、因果效应或商业可用性。

端点可接收调用方提供的 `current_nightly_price_anchor_usd`，但该值只在 raw model path 生成后施加统一 `log1p` 偏移，并标为 `caller_supplied_price_anchor`；它从不进入特征，也不能单凭请求字段自称已验证。酒店详情页只有在严格酒店报价成功返回后才会传入这个锚点。锚定后的路径仍是模型展示，区间不再保留原始经验覆盖保证，也不替代可订报价。模型给出的住宿总额只是 `estimated nightly × stay_nights` 的演示派生值，不代表供应商含税总价或一次性费用。

### 4.4 有条件暂缓

| 候选 | 何时才适合 |
| --- | --- |
| 官方节假日、展会、演唱会、会议 | 有版本化事件身份、地点、开始/结束及 `published_at`；取消/改期也按 as-of 处理 |
| 滞后城市市场中位价或相似酒店价格指数 | 只用预测时刻之前的快照；同一时刻横截面必须 leave-one-property-out，不能包含当前标签 |
| 酒店/城市入住率、剩余库存、预订速度 | 训练与服务都能得到同口径实时或滞后数据；最终入住率绝不回填早期预测 |
| 房型、床型、早餐、退改、官方直订、销售渠道 | 建立单独的房型/费率级目标，且推理时确实会提供这些产品属性 |
| 品牌/连锁 | 有稳定且带有效日期的品牌映射，并在新品牌/独立酒店上验证泛化 |
| 周边酒店、交通与景点密度 | OSM 查询覆盖完整或带覆盖率；部分 Overpass 返回数不能冒充真实供给总量 |
| 文本评价情感 | 有合法、同一物业身份、严格截至预测时刻的文本快照，并证明超过评分/评论数基线的增益 |

### 4.5 明确拒绝

| 因素 | 拒绝原因 |
| --- | --- |
| 当前 `nightly_price`、`total_price`、税前价 | 就是标签或其直接变体 |
| `price_source`、最低价胜出平台、`special_offer`/折扣标签 | 往往由同次价格比较决定，预测前未知或与标签循环 |
| `hotel_id`、物业名称直接 one-hot | 容易记忆反复出现的酒店；对新酒店泛化差，也会让随机/重复样本切分显得虚高 |
| `property_token`、search ID、预订 URL | 短期、敏感或仅用于供应商流程，不是价格因果特征 |
| 今天抓到的评分、评论数、星级回填多年前报价 | 把未来信息带回历史训练行 |
| 最终入住率、最终预订量、入住后评价 | 发生在预测之后 |
| 当前天气或入住日实际天气 | 对未来住宿时刻不可知；长提前期也超出可靠预报范围 |
| 用户点击、搜索次数、设备或个人画像 | 当前项目没有合法、稳定且必要的数据；还会增加隐私与选择偏差风险 |

## 5. As-of 与泄漏边界

### 5.1 每条外部特征的最小元数据

凡是来自机场目录、市场统计、新闻、日历、活动、评分或供应商的外部字段，至少保存：

- `source` 与来源版本/文件哈希；
- `observed_at`：系统实际取得该值的时间；
- `available_at` 或 `published_at`：该值最早可用于预测的时间；
- `source_period_start` / `source_period_end`：统计覆盖期；
- `expires_at` 或 `context_age_days`：适用性与陈旧度；
- 覆盖状态与缺失原因。

训练行只允许连接满足以下条件的记录：

```text
feature.available_at <= prediction_time
feature.observed_at  <= prediction_time
```

若来源后来修订历史值，应保留预测当时可见版本；只有任务明确评估“使用最终修订数据的回溯模型”时才可另建数据集，且不能把结果称为在线 as-of 性能。

### 5.2 聚合与编码规则

- 所有均值、中位数、市场份额、评论词表、类别频率和标准化参数只能在训练切分上拟合。
- 滚动价格或需求统计必须先按时间排序，再对当前行执行严格左闭/右开的历史窗口；当前标签不能进入自己的聚合。
- 同一酒店、航班、航线日期或供应商查询的重复行应先去重；无法去重时按实体/查询组切分，防止同一证据跨集合复制。
- 目标编码若未来采用，必须在训练折内计算；当前简单 one-hot/显式类别优先。
- 缺失是信息，但不能伪装成真实的零、最低星级、国内航线或不可取消。使用 `unknown`、缺失指示或 core 模型。
- 新闻、评分、评论、活动和库存等会变化的字段必须是历史快照；静态机场坐标也要记录目录版本。

### 5.3 典型泄漏检查

在训练前执行并记录：

1. 特征列中不存在价格目标、总价、价格排名、价格等级或它们的别名；
2. 不存在 `available_at > prediction_time`；
3. 不存在同一 `source_record_id` 跨训练/校准/测试重复；
4. 滞后特征的最大源时间严格早于当前行；
5. 评分、评论、新闻、节假日/事件和市场统计没有用当前下载值回填全部历史；
6. 所有 imputer、encoder、scaler 和特征选择器仅在训练数据上拟合；
7. 真实报价锚点只在模型输出后用于明确标注的曲线校准，不进入 raw model prediction。

## 6. 模型与评估门槛

### 6.1 建模方法

机票继续使用对 `log1p(price_usd)` 拟合的直方图梯度提升回归。酒店使用同一简单框架，对 `log1p(nightly_price_usd)` 拟合；它能处理非线性的提前期、入住日历、品质和位置交互，而无需引入更复杂的深度模型。

酒店训练 core 与 enriched 两个候选，并已实现 `train -> selection -> conformal calibration -> test` 四段时间顺序：两个候选使用相同 train，只有 enriched 在 selection 的 MAE 至少改善预注册的 US$0.50 实际门槛时才选 enriched，否则保留 core；独立的后一段使用 `log1p` 绝对残差校准 80% 区间，再反变换为非负、通常不对称的美元区间；最终 test 在所有因素确定前保持不可见。同一时间戳的报价快照不跨段拆分。机票也采用同一四段顺序，但其新增因素选择还受上文预注册实际改善、多候选校正与 bootstrap 门槛约束，且区间在美元绝对残差空间校准。

scikit-learn 的 [TimeSeriesSplit 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 也明确指出，时间有序数据不应使用会导致“未来训练、过去测试”的普通随机切分。

### 6.2 新因素的最低验收门槛

每次新增因素都必须输出与未新增版本相同测试行上的结果：

1. **主要点预测指标**：MAE；同时报告 RMSE、R² 和训练期中位数基线 MAE。
2. **配对比较**：对同一测试行的绝对误差差值做配对 bootstrap 或等价的不确定性分析；不能只比较四舍五入后的单个均值。
3. **区间指标**：80% 区间经验覆盖率与平均宽度；若点误差小幅下降却导致明显欠覆盖，不接受。
4. **分组稳定性**：机票至少按国内/国际、舱位、已见/未见航线；酒店至少按目的地、物业类型、提前期、星级缺失与否检查 MAE 和样本量。
5. **覆盖与陈旧度**：报告每个外部因素的非缺失率、`unknown` 率、来源年龄与回退模型使用比例。
6. **漂移检查**：按时间画残差与覆盖；不能仅用全期平均掩盖后段退化。
7. **简单性优先**：若增强模型没有稳定降低选择段 MAE，或最终时间外测试不支持改善，保留 core；“论文认为有关”本身不构成上线理由。

对于合成演示，验收范围仅是：数据生成可复现、目标未泄漏、时间切分正确、模型优于合成中位数基线、序列化/服务一致、区间计算有效。README、模型卡和 API 都必须继续标注 synthetic，不得把合成 MAE、R² 或覆盖率表述为真实市场性能。

## 7. 数据来源边界

| 来源 | 可支持因素 | 关键限制 |
| --- | --- | --- |
| [BTS O&D Product DB1B/DB1C](https://www.bts.gov/topics/airlines-and-airports/origin-and-destination-survey-data-product) | 已售票金额、行程、承运人、市场、乘客权重 | 抽样已售票，不是实时搜索快照；通常没有准确询价提前期；票价产品口径必须核对 |
| [BTS T-100](https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=Nv4+Pn44vr45&gnoyr_VQ=GDM) | 月度运力、班次、旅客、载客率、承运人竞争 | 有发布滞后和覆盖边界；只能 as-of 使用最后已发布期 |
| [EIA 航空煤油价格](https://www.eia.gov/dnav/pet/hist/eer_epjk_pf4_rgc_dpgW.htm) | 美国区域滞后燃油成本 | 不是全球逐航班成本；存在传导时滞和航司异质性 |
| [OurAirports](https://ourairports.com/data/) | 机场国家、类型、经纬度 | 公共领域但无准确性保证；需版本/哈希与未知回退 |
| [SerpApi Google Flights Price Insights](https://serpapi.com/google-flights-price-insights) | 独立展示的市场历史上下文 | 需要密钥/额度；同次搜索的洞察不能作为同次价格标签的特征 |
| [SerpApi Google Hotels](https://serpapi.com/google-hotels-api) | 真实物业价、日期、住客、坐标、星级、评分、评论、设施、取消 | 需要密钥/额度；取到增强因素时通常也已取到标签；不得自动批量制造训练集 |
| [SearchAPI.io Google Hotels](https://www.searchapi.io/docs/google-hotels-api) | 与上类似，另有位置/交通/机场可达性字段 | 需要密钥且本项目有严格本地额度；同样遵守 as-of 与不可拆分价格证据 |
| OSM / Nominatim / Overpass | 物业类型、坐标、城市中心和周边实体 | 公共服务有限流和覆盖缺口；部分返回数量不是完整市场供给，不能批量滥用 |

任何真实数据接入还必须遵守来源最新条款、缓存/再分发规则和项目的凭据隔离要求。API key、短期 token、完整供应商请求 URL、`.env` 与原始私有数据均不得进入仓库、训练产物或公开日志。
