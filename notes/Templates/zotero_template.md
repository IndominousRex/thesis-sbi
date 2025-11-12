---
title: "{{title}}"
{% if date %}year: {{date | format("YYYY")}}{% endif %}
authors: {{authors}}{{directors}}
{% if citationKey %}citekey: {{citationKey}}{% endif %}
tags:
  - zotero
Optional: store related links for Dataview queries too
related:
  - 
---

{{ bibliography }}

- zotero: [{{desktopURI}}]({{desktopURI}})
- url: {{url}}
{% if pdfLink -%}
- pdf: {{pdfLink}}
{%- endif -%}

{% if abstractNote %}
# Abstract
{{ abstractNote }}
{% endif %}

# Highlights
{% for annotation in annotations %}
{% if annotation.annotatedText %}
> [!cite]
> {{annotation.annotatedText}}
> [Page {{annotation.page}}](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.page}})
{%- if annotation.comment %}
> > [!note]
> > {{annotation.comment}}
{% endif %}
{% else %}
{% if annotation.comment %}
> [!note]
> {{annotation.comment}}
> [Page {{annotation.page}}](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.page}})
{% endif -%}
{% endif -%}
{% if annotation.imageRelativePath %}
![[{{annotation.imageRelativePath}}]]
{% endif -%}
{% endfor %}

# Related
{% persist "related" %}
- [[Paste another paper title or citekey note here]]
{% endpersist %}

# Notes
{% persist "notes" %}
{% endpersist %}

## Suggested (by overlapping tags)
```dataview
LIST
FROM "02_Literature_Notes"
WHERE file.path != this.file.path
AND length(intersection(file.tags, this.file.tags)) > 0
SORT file.mtime desc
LIMIT 10