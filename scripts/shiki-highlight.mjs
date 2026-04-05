import he from 'he'
import { stderr as error, stdin as input, stdout as output } from 'node:process'
import { createHighlighter } from 'shiki'

const THEME = 'material-theme-ocean'
const CODE_BACKGROUND = '#00010A'
const CODE_FOREGROUND = '#F8F8F2'
const { decode } = he

const LANGUAGE_ALIASES = new Map([
    ['plaintext', 'text'],
    ['plain', 'text'],
    ['text', 'text'],
    ['shell', 'bash'],
    ['sh', 'bash'],
    ['zsh', 'bash'],
    ['yml', 'yaml'],
    ['vimscript', 'vim'],
    ['dockerfile', 'docker'],
])

const CODE_BLOCK_PATTERN = /(?:<div class="language-([a-z0-9_+-]+) highlighter-rouge"><div class="highlight">)?<pre(?:\s[^>]*)?>\s*<code(?:\s[^>]*)?(?:class="([^"]*)")?(?:\s[^>]*)?>([\s\S]*?)<\/code>\s*<\/pre>(?:<\/div><\/div>)?/g

let highlighterPromise

function getLanguage(rougeLanguage, className = '') {
    if (rougeLanguage) {
        const normalized = rougeLanguage.toLowerCase()
        return LANGUAGE_ALIASES.get(normalized) ?? normalized
    }

    const match = className.match(/(?:language|lang)-([a-z0-9_+-]+)/i)
    if (!match) {
        return 'text'
    }

    const normalized = match[1].toLowerCase()
    return LANGUAGE_ALIASES.get(normalized) ?? normalized
}

function hasExplicitLanguage(rougeLanguage, className = '') {
    if (rougeLanguage) {
        return true
    }

    return /(?:language|lang)-([a-z0-9_+-]+)/i.test(className)
}

async function getHighlighter() {
    if (!highlighterPromise) {
        highlighterPromise = createHighlighter({
            themes: [THEME],
            langs: ['text'],
        })
    }

    return highlighterPromise
}

async function highlightBlock(highlighter, rougeLanguage, className, encodedCode, previousBlockLanguage, fallbackLanguage) {
    const detectedLanguage = getLanguage(rougeLanguage, className)
    const kind = detectedLanguage === 'text' && previousBlockLanguage && previousBlockLanguage !== 'text'
        ? 'output'
        : 'code'
    const language = kind === 'code' && detectedLanguage === 'text' && fallbackLanguage
        ? fallbackLanguage
        : detectedLanguage
    const normalizedCode = rougeLanguage
        ? encodedCode.replace(/<\/?span[^>]*>/g, '')
        : encodedCode
    const code = decode(normalizedCode)
    const buttonLabel = kind === 'output'
        ? 'Copy output'
        : language === 'text'
            ? 'Copy code'
            : `Copy ${language}`

    try {
        await highlighter.loadLanguage(language)
        return wrapHighlightedBlock(
            highlighter.codeToHtml(code, {
                lang: language,
                theme: THEME,
            }),
            { kind, lang: language, label: buttonLabel },
        )
    } catch {
        return wrapHighlightedBlock(
            highlighter.codeToHtml(code, {
                lang: 'text',
                theme: THEME,
            }),
            { kind, lang: 'text', label: kind === 'output' ? 'Copy output' : 'Copy code' },
        )
    }
}

function wrapHighlightedBlock(highlightedHtml, { kind, lang, label }) {
    const withThemeOverrides = highlightedHtml
        .replace(/background-color:[^;]+;/, `background-color:${CODE_BACKGROUND};`)
        .replace(/color:#babed8/g, `color:${CODE_FOREGROUND}`)
        .replace(/color:#BABED8/g, `color:${CODE_FOREGROUND}`)

    return [
        `<div class="code-block" data-code-block data-code-kind="${kind}" data-code-language="${lang}">`,
        `<button class="code-block__copy" type="button" data-code-copy aria-label="${label}" title="${label}"></button>`,
        withThemeOverrides,
        '</div>',
    ].join('')
}

async function readStdin() {
    const chunks = []

    for await (const chunk of input) {
        chunks.push(chunk)
    }

    return Buffer.concat(chunks).toString('utf8')
}

async function main() {
    const rawInput = await readStdin()
    const payload = JSON.parse(rawInput)
    const html = payload.html ?? ''
    const languages = Array.isArray(payload.languages) ? [...payload.languages] : []
    if (!html.includes('<pre') || !html.includes('<code')) {
        output.write(html)
        return
    }

    const highlighter = await getHighlighter()
    const matches = Array.from(html.matchAll(CODE_BLOCK_PATTERN))

    if (matches.length === 0) {
        output.write(html)
        return
    }

    let lastIndex = 0
    const highlighted = []
    let previousBlockLanguage = null

    for (const match of matches) {
        const [fullMatch, rougeLanguage, className = '', encodedCode] = match
        const detectedLanguage = getLanguage(rougeLanguage, className)
        const fallbackLanguage = detectedLanguage === 'text' && (!previousBlockLanguage || previousBlockLanguage === 'text')
            ? languages.shift()
            : undefined
        highlighted.push(html.slice(lastIndex, match.index))
        const highlightedBlock = await highlightBlock(
            highlighter,
            rougeLanguage,
            className,
            encodedCode,
            previousBlockLanguage,
            fallbackLanguage,
        )
        highlighted.push(highlightedBlock)
        previousBlockLanguage = detectedLanguage === 'text' && fallbackLanguage ? fallbackLanguage : detectedLanguage
        lastIndex = match.index + fullMatch.length
    }

    highlighted.push(html.slice(lastIndex))
    output.write(highlighted.join(''))
}

main().catch((cause) => {
    error.write(`${cause instanceof Error ? cause.stack : String(cause)}\n`)
    process.exitCode = 1
})