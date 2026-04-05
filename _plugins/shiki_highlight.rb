require "open3"
require "json"
require "cgi"

module AOLabs
  module ShikiHighlight
    module_function

    MERMAID_FENCE_PATTERN = /^([ \t]*)```mermaid[^\n]*\n(.*?)^\1```[ \t]*\n?/m
    MERMAID_PLACEHOLDER_PREFIX = "AOLABS_MERMAID_BLOCK_"

    def transform(site, html, source_content)
      script_path = File.join(site.source, "scripts", "shiki-highlight.mjs")
      payload = {
        html: html,
        languages: extract_fence_languages(source_content),
      }
      stdout, stderr, status = Open3.capture3(
        "node",
        script_path,
        stdin_data: JSON.generate(payload),
        chdir: site.source,
      )

      return stdout if status.success?

      raise Jekyll::Errors::FatalException, <<~ERROR
        Shiki build-time highlighting failed.
        Run `npm install` to install the Node dependencies for this site.
        #{stderr.strip}
      ERROR
    end

    def extract_fence_languages(content)
      content.scan(/^\s*```\s*([^\s`]+)?/).flatten.compact.map(&:downcase)
    end

    def protect_mermaid_blocks(content)
      mermaid_blocks = []

      protected_content = content.gsub(MERMAID_FENCE_PATTERN) do
        index = mermaid_blocks.length
        mermaid_blocks << Regexp.last_match(2)
        mermaid_placeholder(index)
      end

      [protected_content, mermaid_blocks]
    end

    def restore_mermaid_blocks(html, mermaid_blocks)
      mermaid_blocks.each_with_index.reduce(html) do |result, (code, index)|
        result.gsub(mermaid_placeholder(index), mermaid_html(code))
      end
    end

    def mermaid_placeholder(index)
      %(<mermaid-placeholder data-mermaid-block="#{MERMAID_PLACEHOLDER_PREFIX}#{index}"></mermaid-placeholder>)
    end

    def mermaid_html(code)
      %(<pre><code class="language-mermaid">#{CGI.escapeHTML(code)}</code></pre>)
    end
  end
end

Jekyll::Hooks.register [:pages, :documents], :pre_render do |document|
  next unless document.output_ext == ".html"

  protected_content, mermaid_blocks = AOLabs::ShikiHighlight.protect_mermaid_blocks(document.content.to_s)
  document.content = protected_content
  document.data["aolabs_mermaid_blocks"] = mermaid_blocks
  document.data["aolabs_shiki_source_content"] = protected_content
end

Jekyll::Hooks.register [:pages, :documents], :post_render do |document|
  next unless document.output_ext == ".html"

  output = document.output
  source_content = document.data["aolabs_shiki_source_content"].to_s

  if output.include?("<pre") && output.include?("<code")
    output = AOLabs::ShikiHighlight.transform(document.site, output, source_content)
  end

  mermaid_blocks = Array(document.data["aolabs_mermaid_blocks"])
  output = AOLabs::ShikiHighlight.restore_mermaid_blocks(output, mermaid_blocks) if mermaid_blocks.any?

  document.output = output
end