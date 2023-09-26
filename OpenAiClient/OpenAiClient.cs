namespace Ckeisc.OpenAi;

using System.Net.Http.Headers;
using System.Runtime.Serialization;
using System.Text.Json;
using System.Text.Json.Serialization;

public class OpenAiClient
{
    private const string OpenAiChatApiBase = "https://api.openai.com/v1/chat/completions";
    private const string OpenAiImageGenerationApiBase = "https://api.openai.com/v1/images/generations";
    private const string OpenAiImageEditApiBase = "https://api.openai.com/v1/images/edits";

    private HttpClient client;
    private JsonSerializerOptions jsonOptions;

    private Dictionary<ChatModels, string> modelNames = new() {
        { ChatModels.Gpt3_5Turbo, "gpt-3.5-turbo" },
        { ChatModels.Gpt3_5Turbo16k, "gpt-3.5-turbo-16k" }
    };

    /// <summary>
    /// OpenAI client
    /// </summary>
    /// <param name="apiKey"></param>
    /// <param name="httpClientHandler">for unit test</param>
    public OpenAiClient(string apiKey, HttpMessageHandler? unitTestHandler = null)
    {
        this.client = (unitTestHandler != null) ? 
            new HttpClient(unitTestHandler) : new HttpClient();
        this.client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", apiKey);
        this.client.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("application/json"));
        // for streaming response
        this.client.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("text/event-stream"));
        this.jsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            Converters =
            {
                new JsonStringEnumMemberConverter(JsonNamingPolicy.CamelCase)
            }
        };
    }

    public async IAsyncEnumerable<ChatCompletionChunk> StreamChat(
        IEnumerable<ChatMessage> messages, 
        ChatModels chatModel = ChatModels.Gpt3_5Turbo,
        int maxTokens = 256)
    {
        ChatCreateRequest request = this.CreateChatCreateRequest(messages, chatModel, maxTokens, isStream: true);
        using StringContent stringContent = this.SerializeStringContent(request);
        using HttpResponseMessage responseMessage = await this.client.PostAsync(
            OpenAiChatApiBase, stringContent);
        responseMessage.EnsureSuccessStatusCode();

        using Stream stream = await responseMessage.Content.ReadAsStreamAsync();
        using StreamReader reader = new StreamReader(stream);

        string? line;
        while ((line = await reader.ReadLineAsync()) != null)
        {
            if (line.StartsWith(":"))
            {
                // comment line starts with ":"
                continue;
            }
            else if (line == "data: [DONE]")
            {
                yield break;
            }
            else if (line.StartsWith("data: {"))
            {
                string json = line.Substring(6);
                ChatCompletionChunk? chunk =
                    JsonSerializer.Deserialize<ChatCompletionChunk>(json, jsonOptions);
                if (chunk == null)
                {
                    throw new InvalidDataException("failed to parse response");
                }
                yield return chunk;
            }
        }

        yield break;
    }

    public async Task<ChatCompletionResponse> CreateChat(
        IEnumerable<ChatMessage> messages,
        ChatModels chatModel = ChatModels.Gpt3_5Turbo,
        int maxTokens = 256)
    {
        ChatCreateRequest request = this.CreateChatCreateRequest(messages, chatModel, maxTokens, isStream: false);
        using StringContent stringContent = this.SerializeStringContent(request);
        using HttpResponseMessage responseMessage = 
            await this.client.PostAsync(OpenAiChatApiBase, stringContent);
        responseMessage.EnsureSuccessStatusCode();
        ChatCompletionResponse response = 
            await this.DeserializePayload<ChatCompletionResponse>(responseMessage);
        return response;
    }

    public async Task<ImageResponse> CreateImage(
        string prompt, ImageResponseFormat format = ImageResponseFormat.Url)
    {
        ImageGenerationRequest request = this.CreateImageRequest(prompt, format);
        using StringContent stringContent = this.SerializeStringContent(request);
        using HttpResponseMessage responseMessage = await this.client.PostAsync(
            OpenAiImageGenerationApiBase, stringContent);
        responseMessage.EnsureSuccessStatusCode();
        ImageResponse response =
            await this.DeserializePayload<ImageResponse>(responseMessage);
        return response;
    }

    public async Task<ImageResponse> EditImage(
        string prompt, string imagePath, string maskPath, ImageResponseFormat format = ImageResponseFormat.Url)
    {
        string formatString = format == ImageResponseFormat.Url ? "url" : "b64_json";

        using FileStream imageStream = File.OpenRead(imagePath);
        using FileStream maskStream = File.OpenRead(maskPath);

        using MultipartFormDataContent content = new ()
        {
            { new StreamContent(imageStream), "image", "image.png" },
            { new StreamContent(maskStream), "mask", "mask.png" },
            { new StringContent(prompt), "prompt" },
            { new StringContent("1"), "n" },
            { new StringContent("512x512"), "size" },
            { new StringContent(formatString), "response_format" }
        };

        using HttpResponseMessage responseMessage = await this.client.PostAsync(
            OpenAiImageEditApiBase, content);
        responseMessage.EnsureSuccessStatusCode();
        ImageResponse response =
            await this.DeserializePayload<ImageResponse>(responseMessage);
        return response;
    }

    private ChatCreateRequest CreateChatCreateRequest(
        IEnumerable<ChatMessage> messages, ChatModels chatModel, int maxTokens, bool isStream)
    {
        ChatCreateRequest request = new()
        {
            Model = this.modelNames[chatModel],
            Messages = messages.ToArray(),
            Stream = isStream,
            Temperature = 1F,
            MaxTokens = maxTokens,
            NucleusSamplingFactor = 1F,
            FrequencyPenalty = 0,
            PresencePenalty = 0
        };
        return request;
    }

    private ImageGenerationRequest CreateImageRequest(
        string prompt, ImageResponseFormat format)
    {
        return new ImageGenerationRequest()
        {
            Prompt = prompt,
            Count = 1,
            Size = ImageSize.Square512,
            ResponseFormat = format
        };
    }

    private StringContent SerializeStringContent(object payload)
    {
        string jsonString = JsonSerializer.Serialize(payload, this.jsonOptions);
        StringContent stringContent = new StringContent(
            jsonString, new MediaTypeHeaderValue("application/json"));
        return stringContent;
    }

    private async Task<T> DeserializePayload<T>(HttpResponseMessage responseMessage)
    {
        string json = await responseMessage.Content.ReadAsStringAsync();
        T? response =
            JsonSerializer.Deserialize<T>(json, this.jsonOptions);

        if (response == null)
        {
            throw new InvalidDataException("missing or failed to parse response");
        }

        return response;
    }
}

public class ImageGenerationRequest
{
    public required string Prompt { get; set; }

    [JsonPropertyName("n")]
    public int Count { get; set; }

    public ImageSize Size { get; set; }

    [JsonPropertyName("response_format")]
    public ImageResponseFormat ResponseFormat { get; set; }
}

public class ImageResponse
{
    public required UrlData[] Data { get; set; }
}

public class UrlData
{
    public string? Url { get; set; }

    [JsonPropertyName("b64_json")]
    public string? Base64Json { get; set; }
}

public class ChatCreateRequest
{
    public required string Model { get; set; }

    public required ChatMessage[] Messages { get; set; }

    public bool Stream { get; set; }

    public float Temperature { get; set; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; set; }

    [JsonPropertyName("top_p")]
    public float NucleusSamplingFactor { get; set; }

    [JsonPropertyName("frequency_penalty")]
    public int FrequencyPenalty { get; set; }

    [JsonPropertyName("presence_penalty")]
    public int PresencePenalty { get; set; }
}

public enum ChatRole
{
    System,
    User,
    Assistant
}

public enum ChatModels
{
    Gpt3_5Turbo,
    Gpt3_5Turbo16k
}

public enum FinishReason
{
    Stop,
    Length,
    [EnumMember(Value = "function_call")]
    FunctionCall,
    [EnumMember(Value = "content_filter")]
    ContentFilter
}

public enum ImageSize
{
    [EnumMember(Value = "256x256")]
    Square256,
    [EnumMember(Value = "512x512")]
    Square512,
    [EnumMember(Value = "1024x1024")]
    Square1024
}

public enum ImageResponseFormat
{
    Url,
    [EnumMember(Value = "b64_json")]
    Base64Json
}

public class ChatCompletionResponse
{
    public required string Id { get; set; }

    public required string Model { get; set; }

    public required ChatCompletionChoice[] Choices { get; set; }
}

public class ChatCompletionChunk
{
    public required string Id { get; set; }

    public required string Model { get; set; }

    public required ChatCompletionDelta[] Choices { get; set; }
}

public class ChatMessage
{
    public ChatRole Role { get; set; }

    public string Content { get; set; } = string.Empty;
}

public class ChatCompletionChoice
{
    public int Index { get; set; }

    public required ChatMessage Message { get; set; }

    [JsonPropertyName("finish_reason")]
    public FinishReason? FinishReason { get; set; }
}

public class ChatCompletionDelta
{
    public int Index { get; set; }

    public required ChatMessage Delta { get; set; }

    [JsonPropertyName("finish_reason")]
    public FinishReason? FinishReason { get; set; }
}
