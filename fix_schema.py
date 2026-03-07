content = open('src/extraction/schemas.py', encoding='utf-8').read()

old1 = 'dietary_tags: Optional[list[str]] = Field(\n        default=None,\n        description="Canonical dietary tags from controlled vocabulary"'
new1 = 'dietary_tags: list[str] = Field(\n        default_factory=list,\n        description="Canonical dietary tags from controlled vocabulary"'

old2 = 'allergen_list: Optional[list[str]] = Field(\n        default=None,\n        description="Allergens present from controlled vocabulary"'
new2 = 'allergen_list: list[str] = Field(\n        default_factory=list,\n        description="Allergens present from controlled vocabulary"'

content = content.replace(old1, new1).replace(old2, new2)
open('src/extraction/schemas.py', 'w', encoding='utf-8').write(content)
print('Done')
