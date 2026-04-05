require "open3"
require "json"

module AOLabs
  module ShikiHighlight
    module_function

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
  end
end

Jekyll::Hooks.register [:pages, :documents], :post_render do |document|
  next unless document.output_ext == ".html"
  next unless document.output.include?("<pre") && document.output.include?("<code")

  source_content = begin
    File.read(document.path)
  rescue StandardError
    document.content
  end

  document.output = AOLabs::ShikiHighlight.transform(document.site, document.output, source_content)
end