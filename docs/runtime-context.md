# Runtime context and free-data fallbacks

Flight Forecast Lab can enrich its synthetic demonstration models with short-lived external
context. External data is never presented as part of the training corpus and is never silently
substituted for a live bookable fare.

## Resolution order

| Signal | Preferred source | Fallback | Meaning in the response |
| --- | --- | --- | --- |
| Weather | Open-Meteo current/forecast plus NOAA METAR/TAF | Clearly labelled synthetic month/latitude prior | `live`, `forecast`, or `proxy` |
| Airport operations | AirLabs schedules or ADSB.lol near the departure time | Clearly labelled synthetic airport/time prior | `live` or `proxy` |
| Disruption news | No-key GDELT DOC 2.0 articles from the recent seven-day window, ordered `DateDesc`; official GAL rolling RSS if DOC fails | Route cache no older than six hours with reduced influence, otherwise a neutral value with no articles | `live`, `historical`, or `neutral` |
| Route airlines | AirLabs route records when a key is configured | Global connecting model-scenario catalog | `provider_confirmed` or `model_scenario` |

The service catches provider timeouts, malformed payloads, quota exhaustion, and empty results.
It then returns a labelled fallback rather than failing the whole prediction. Set
`EXTERNAL_CONTEXT_ENABLED=0` for deterministic offline development.

Within two hours of departure, the service uses fresh Open-Meteo current model conditions and
can blend NOAA METAR airport observations with TAF. From two to 30 hours it blends the
departure-hour Open-Meteo forecast with TAF; after that, it uses Open-Meteo hourly forecasts up
to the provider's 16-day limit. Open-Meteo current conditions are based on 15-minute model data,
not an airport sensor observation. If Open-Meteo is unavailable near departure, a fresh METAR
can now serve as the independent `live` fallback. NOAA responses are cached for five minutes to
stay below the provider's per-thread frequency guidance. TAF risk is calculated only from decoded
forecast segments that cover the requested departure time; METAR risk also reads structured wind,
gust, visibility, flight category, and ceiling fields rather than relying on weather keywords alone.

Current AirLabs schedules and
ADSB.lol density are only used within six hours; later departures receive `proxy` averages
computed from training rows only (the demo source is `synthetic_demo_training_average`).
The demo has no validated historical aggregate, so it never labels those hand-built priors as
`historical`. A production deployment may use that status only after loading audited historical
averages.

The dashboard sends a timezone-free wall time. The service resolves the origin airport from its
coordinates, determines the IANA timezone offline, handles daylight-saving offsets, and returns
both the normalized timestamp and timezone. API clients may alternatively send an aware ISO 8601
timestamp, which is treated as an absolute instant.

AirLabs route matches are treated as confirmed direct carriers. Every other catalog carrier is
kept as a one-stop model scenario so “direct first” never labels an unconfirmed route as direct.
Cabins always remain `catalog_scenario`, because the route source does not confirm cabin inventory.
For a one-stop scenario the displayed duration adds a 90-minute connection and the itinerary
on-time probability is the direct-leg probability squared. The response labels this assumption as
`two_leg_independence_scenario`; it is not a claim about a real connection or transfer airport.

## News feature

Recent news is queried without an API key through GDELT DOC 2.0, using the origin and destination
codes/names together with a narrow disruption vocabulary covering airport closure, strikes,
conflict, severe disruption, cancellations, and related events. The seven-day query requests
`DateDesc`, so results are processed newest-observed first. GDELT's global source stream is
near-real-time rather than instantaneous; its core data updates on an approximately 15-minute
cycle.

If the DOC request fails, the service tries GDELT's official Global Article List (GAL) RSS feed.
That feed updates every minute and contains a rolling window of roughly the latest 15 minutes.
Only route-relevant disruption titles are accepted from the global RSS stream. Successful results
are cached by route for 15 minutes, independent of the requested departure date, which avoids
repeating the same external query and reduces rate-limit pressure. Concurrent requests for the
same route share a single refresh. If both live paths fail, a successful route cache no older than
six hours may be returned as `historical`; its risk contribution is reduced. With no usable cache,
the response is `neutral`, has value zero, and contains no articles.

Only real returned article titles are scored. The bounded score becomes `news_disruption_index`,
which is a feature in both demo models. Current news is less informative for departures farther in
the future, so its model influence is attenuated as lead time increases. Articles are deduplicated
by canonical URL and normalized title, and results prefer independent source domains instead of
counting syndicated copies as separate corroboration.

This is deliberately conservative:

- no article is invented when GDELT and the bounded stale cache are unavailable;
- article presence does not prove that a particular flight will be affected;
- the UI exposes article links, source, optional source language, observed/indexed time, and the
  signal status;
- GDELT's article time usually means when GDELT observed/indexed the URL, not a guaranteed exact
  publisher timestamp; headlines remain in their source language;
- the synthetic training relationship is an engineering demonstration, not validated causal
  evidence that news raises a particular fare or delay probability.

## Student, baggage, and fare-rule fields

The comparison response uses three-state or multi-state policy fields. `unknown` is not treated as
`not_included`. A public airline student-program page is labelled `program_available`; it is not
labelled as an actual student fare for the requested trip. `confirmed_free` is reserved for
offer-level evidence, which the strict-free model-estimate mode normally does not have.

The student ranking follows the requested lexicographic order:

1. lower estimated price;
2. confirmed free checked baggage;
3. confirmed actual student discount;
4. confirmed free change/refund;
5. lower age and verification burden.

`program_available` alone does not satisfy the actual-discount criterion. Its published age and
verification metadata is considered only in the fifth criterion. Because estimated prices will
usually differ, later criteria most often act as tie-breakers.

## Source links and operational limits

- [NOAA Aviation Weather Data API](https://aviationweather.gov/data/api/)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Open-Meteo free API terms and attribution](https://open-meteo.com/en/terms)
- [AirLabs schedules API](https://airlabs.co/docs/schedules)
- [ADSB.lol public API](https://api.adsb.lol/)
- [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [GDELT Global Article List RSS](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/)
- [GDELT project use and attribution](https://www.gdeltproject.org/about.html)
- [OurAirports public-domain data](https://ourairports.com/data/)

Free-provider quotas, fields, coverage, and terms can change. Review the linked sources before
deploying publicly or commercially. Use or redistribution of GDELT data must cite and link the
GDELT Project; the dashboard includes that visible attribution. API keys belong in `.env`, never
in the repository.
