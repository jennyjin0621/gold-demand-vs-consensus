# Manual downloads (git-ignored)

Most data loads automatically. These are the fallbacks and the one manual source.

| File | When you need it | Where |
|---|---|---|
| `sge_au9999.csv` | only if akshare breaks | sge.com.cn daily market data, Au99.99. Columns `date, close` (CNY per gram) |
| `gld_archive.csv` | only if the SPDR URL moves | spdrgoldshares.com, GLD historical archive |
| WGC central bank stats (.xlsx) | Step 2 onward | gold.org/goldhub/data/monthly-central-bank-statistics (free account). Load with `data.wgc_table(path, header_row=...)` |
| `articles/*.txt` | Step 3 (consensus) | one news article per file, URL on the first line, publish date on the second |
