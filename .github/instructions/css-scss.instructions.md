---
description:
  "Use when modifying styles, colors, layout, or any CSS/SCSS changes on this
  Jekyll site. Covers the Material Ocean palette, Sass module structure,
  component/layout files, and the shared theme helpers."
applyTo: "_sass/**,assets/css/main.scss"
---

# CSS / SCSS Modification Guidelines

The site now uses Sass modules. Treat `assets/css/main.scss` as the only
entrypoint and keep feature styling in the appropriate `_sass/` partial.

## Directory Structure

```
_sass/
├── libs/
│   ├── _vars.scss        ← color palette, palette() function, breakpoints
│   ├── _index.scss       ← shared module export surface
│   ├── _mixins.scss      ← reusable mixins
│   ├── _shared.scss      ← shared layout helpers such as inner()
│   ├── _functions.scss   ← utility functions
│   └── _skel.scss        ← grid/layout utilities
├── base/
│   ├── _page.scss        ← global page styles
│   └── _typography.scss  ← font, heading, text styles
├── components/
│   ├── _box.scss
│   ├── _button.scss
│   ├── _contact-method.scss
│   ├── _form.scss
│   ├── _icon.scss
│   ├── _image.scss
│   ├── _list.scss
│   ├── _mermaid.scss
│   ├── _search.scss
│   ├── _section.scss
│   ├── _spotlights.scss
│   ├── _table.scss
│   ├── _tags.scss
│   └── _tiles.scss
└── layout/
    ├── _banner.scss
    ├── _contact.scss
    ├── _footer.scss
  ├── _header-nav.scss
  ├── _header-responsive.scss
    ├── _header.scss
    ├── _main.scss
    ├── _menu.scss
  ├── _post.scss
    └── _wrapper.scss
```

## Color Palette — Material Ocean

All colors live in `_sass/libs/_vars.scss` under the `$palette` map. Never use
raw hex values in theme code; always go through `palette()` and related helpers.

| Key         | Hex                       | Usage                                 |
| ----------- | ------------------------- | ------------------------------------- |
| `bg`        | `#0f111a`                 | Main background                       |
| `bg-alt`    | `#181b26`                 | Alternate background (cards, sidebar) |
| `fg`        | `#ffffff`                 | Primary text                          |
| `fg-bold`   | `#ffffff`                 | Bold/heading text                     |
| `fg-light`  | `rgba(244,244,255,0.2)`   | Muted/secondary text                  |
| `border`    | `rgba(212,212,255,0.1)`   | Dividers and borders                  |
| `border-bg` | `rgba(212,212,255,0.035)` | Subtle panel backgrounds              |
| `highlight` | `#9bf1ff`                 | Cyan — primary accent, links, icons   |
| `accent1`   | `#6fc3df`                 | Blue                                  |
| `accent2`   | `#8d82c4`                 | Purple                                |
| `accent3`   | `#ec8d81`                 | Red/coral                             |
| `accent4`   | `#e7b788`                 | Amber/orange                          |
| `accent5`   | `#8ea9e8`                 | Soft blue                             |
| `accent6`   | `#87c5a4`                 | Teal/green                            |

### Using the theme helpers

```scss
@use "../libs" as *;

// Correct
color: palette(highlight);
background-color: palette(bg-alt);
border: 1px solid palette(border);
padding: size(element-margin);
font-family: font(family);

// Wrong: hardcoded color and legacy private helper
color: #9bf1ff;
color: _palette(highlight);
```

## Adding a New Color

1. Add the key-value pair to the `$palette` map in `_sass/libs/_vars.scss`
2. Reference it with `palette(your-key)` throughout the codebase

## Component vs Layout Changes

| Change type                        | Edit file in                     |
| ---------------------------------- | -------------------------------- |
| Post tags and tag archive UI       | `_sass/components/_tags.scss`    |
| Search page UI                     | `_sass/components/_search.scss`  |
| Mermaid viewer presentation        | `_sass/components/_mermaid.scss` |
| Tile cards on home and archives    | `_sass/components/_tiles.scss`   |
| Header layout and nav interactions | `_sass/layout/_header*.scss`     |
| Post meta and floating controls    | `_sass/layout/_post.scss`        |
| Page background and scrollbar      | `_sass/base/_page.scss`          |
| Global typography                  | `_sass/base/_typography.scss`    |

## Build

Jekyll compiles `assets/css/main.scss` automatically during
`bundle exec jekyll serve`. `sass.style: compressed` is set in `_config.yml`.

## Mixins

Use `@use "../libs" as *;` inside partials so shared helpers are available. Load
feature partials from `assets/css/main.scss` with `@include meta.load-css(...)`.
Do not add new standalone CSS files for theme behavior that belongs in the Sass
pipeline.
