"""Jina Reader URL fetcher with SSRF prevention.

read_url(url) validates the URL before making any network request, then
fetches page content via https://r.jina.ai/{url}.

SSRF validation (using Python stdlib `ipaddress` + `validators` PyPI package)
rejects:
  - Non-http/https schemes  (ftp://, file://, data:, etc.)
  - RFC 1918 private ranges  10.x.x.x, 172.16–31.x.x, 192.168.x.x
  - Loopback                 127.x.x.x, ::1
  - Link-local               169.254.x.x, fe80::/10
  - Multicast / reserved ranges
  - Encoded IP variants      (hex, octal, URL-encoded octets)
  - Malformed / unresolvable hostnames

LLM-generated URLs are never passed to the HTTP client without passing
this validation step first.
"""
