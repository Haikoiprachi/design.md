# Eval results

Model: `gemini-3.5-flash` · run 2026-11-12 18:45

## Freshness

Answer accuracy: **11/11**
Search routing accuracy: **11/11**
Time to first token: p50 **6.06s**, p95 **21.84s**
Total time: p50 **6.23s**, p95 **21.92s**

| case | pass | searched | ttft | total | answer |
|---|---|---|---|---|---|
| stable-capital | ✅ | False | 11.91s | 11.91s | The capital of Australia is Canberra. |
| stable-math | ✅ | False | 2.07s | 2.08s | 391 |
| stable-author | ✅ | False | 1.19s | 1.19s | Jane Austen |
| stable-history | ✅ | False | 1.18s | 1.19s | The Eiffel Tower was completed in **1889**. |
| stable-definition | ✅ | False | 1.44s | 1.55s | A hash table is a data structure that maps keys to values for efficient lookup using a hash function to compute an index |
| fresh-ceo | ✅ | True | 3.71s | 3.72s | Satya Nadella is the CEO of Microsoft [2]. |
| fresh-weather | ✅ | True | 6.07s | 6.37s | As of Thursday evening, November 12, 2026, the weather in Tokyo is clear and around 13°C (55°F), following a daytime hig |
| fresh-news | ✅ | True | 9.91s | 10.00s | I am currently unable to fetch live, real-time news headlines. Please check a major news platform like [Reuters](https:/ |
| fresh-bitcoin | ✅ | True | 6.06s | 6.23s | As of today, Bitcoin is trading at approximately $72,800 to $73,000 USD [1][2]. |
| fresh-python | ✅ | True | 8.06s | 8.62s | As of November 2026, the latest stable release of Python is **Python 3.15.2** [1, 2]. |
| fresh-followup | ✅ | True | 21.84s | 21.92s | As of November 2026, the Prime Minister of the United Kingdom is Andy Burnham, who took office on July 20, 2026 [4]. |

## Memory

Answer accuracy: **6/6**
Time to first token: p50 **1.23s**, p95 **1.99s**
Total time: p50 **1.23s**, p95 **2.02s**

| case | pass | searched | ttft | total | answer |
|---|---|---|---|---|---|
| what-yesterday | ✅ | False | 1.30s | 1.30s | You asked about the population of Iceland. |
| when-asked | ✅ | False | 1.12s | 1.13s | You asked about sourdough on Monday, November 9, at 6:44 PM. |
| first-question | ✅ | False | 1.99s | 2.02s | Your very first question was to recommend a sci-fi novel about Mars (asked on Thursday, October 29, 2026). |
| cross-session-fact | ✅ | False | 1.61s | 1.61s | Your name is Riya. |
| earlier-today | ✅ | False | 1.10s | 1.12s | Earlier today, at 6:34 PM, you asked to convert 5 km to miles. |
| no-record | ✅ | False | 1.15s | 1.16s | No, you did not ask about the stock market last week. Your only previous question was about a pasta recipe. |
