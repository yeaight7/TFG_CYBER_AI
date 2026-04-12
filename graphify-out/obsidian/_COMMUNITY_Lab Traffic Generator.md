---
type: community
members: 4
---

# Lab Traffic Generator

**Members:** 4 nodes

## Members
- [[gen_traffic.py]] - code - lab\docker\generator\gen_traffic.py
- [[http_get()]] - code - lab\docker\generator\gen_traffic.py
- [[main()]] - code - lab\docker\generator\gen_traffic.py
- [[tcp_connect()]] - code - lab\docker\generator\gen_traffic.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Lab_Traffic_Generator
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Project Docs and Phase Plan]]

## Top bridge nodes
- [[gen_traffic.py]] - degree 4, connects to 1 community