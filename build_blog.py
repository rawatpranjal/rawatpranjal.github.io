#!/usr/bin/env python3
"""Pull latest Substack posts into index.html between the POSTS markers.
ponytail: stdlib only (urllib + regex), no feedparser dep. Substack RSS is simple
and well-formed enough that a regex pass beats pulling in a parser.
Run locally: python3 build_blog.py   (Substack 403s datacenter IPs, so not in CI)
"""
import re, sys, urllib.request

FEED = "https://pranjalrawat42.substack.com/feed"
MAX_POSTS = 15
HTML = "index.html"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

def strip_cdata(s):
    m = re.search(r"<!\[CDATA\[(.*?)\]\]>", s, re.S)
    return (m.group(1) if m else s).strip()

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def parse(xml):
    posts = []
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", it, re.S)
        l = re.search(r"<link>(.*?)</link>", it, re.S)
        d = re.search(r"<pubDate>(.*?)</pubDate>", it, re.S)
        if not (t and l):
            continue
        title = esc(strip_cdata(t.group(1)))
        link = strip_cdata(l.group(1))
        mon, yr = "", ""
        if d:
            parts = strip_cdata(d.group(1)).split()  # e.g. Fri, 05 Jun 2026 ...
            if len(parts) >= 4:
                mon, yr = parts[2], "'" + parts[3][-2:]
        posts.append((mon, yr, title, link))
    return posts[:MAX_POSTS]

def render(posts):
    rows = []
    for mon, yr, title, link in posts:
        rows.append(f"""    <div class="entry">
      <div class="date"><div class="yr">{yr}</div><div class="tag">{mon}</div></div>
      <div class="body"><h3><a href="{link}">{title}</a></h3></div>
    </div>""")
    return "\n".join(rows)

def main():
    posts = parse(fetch(FEED))
    if not posts:
        sys.exit("no posts parsed — leaving HTML untouched")
    block = "<!--POSTS:start-->\n" + render(posts) + "\n    <!--POSTS:end-->"
    html = open(HTML, encoding="utf-8").read()
    new = re.sub(r"<!--POSTS:start-->.*?<!--POSTS:end-->", lambda _: block, html, flags=re.S)
    if new != html:
        open(HTML, "w", encoding="utf-8").write(new)
        print(f"updated {HTML} with {len(posts)} posts")
    else:
        print("no change")

if __name__ == "__main__":
    main()
