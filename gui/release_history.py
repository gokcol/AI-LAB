"""Group public history for display without rewriting published source entries."""


def grouped(releases):
    groups = {}
    for entry in releases:
        date = entry["date"]
        if date not in groups:
            groups[date] = {"version": entry["version"], "date": date,
                            "title": entry["title"], "changes": list(entry["changes"])}
        else:
            group = groups[date]
            group["title"] = "Learning tools and improvements"
            group["changes"].extend(c for c in entry["changes"] if c not in group["changes"])
    return list(groups.values())
