export default function (pi) {
  pi.registerProvider("ollama", {
    name: "Ollama (local)",
    baseUrl: "http://localhost:11434/v1",
    apiKey: "ollama",
    api: "openai-completions",
    models: [
      {
        id: "qwen3.6:35b-a3b-nvfp4",
        name: "Qwen3.6-35B-A3B nvfp4",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 4096,
        compat: {
          maxTokensField: "max_tokens",
          supportsUsageInStreaming: true,
        },
      },
    ],
  });
}
