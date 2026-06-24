export default function (pi) {
  pi.registerProvider("omlx", {
    name: "oMLX (local)",
    baseUrl: "http://localhost:8000/v1",
    apiKey: process.env.OMLX_API_KEY ?? "1234",
    api: "openai-completions",
    models: [
      {
        id: "Qwen3.6-35B-A3B-4bit",
        name: "Qwen3.6-35B-A3B 4-bit",
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
