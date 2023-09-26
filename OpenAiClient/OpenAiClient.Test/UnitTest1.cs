using Moq.Protected;
using System.Net;

namespace Test
{
    public class Tests
    {
        private HttpRequestMessage lastRequestMessage;

        readonly ChatMessage[] userChatPrompt = new[]
        {
            new ChatMessage()
            {
                Role = ChatRole.User,
                Content = "How are you"
            }
        };

        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public async Task ChatRequestSerialization()
        {
            OpenAiClient client = getTestClient(@"ApiJson\ChatResponse.json");
            await client.CreateChat(this.userChatPrompt);
            Assert.NotNull(this.lastRequestMessage);
            string json = await this.lastRequestMessage.Content.ReadAsStringAsync();
            Assert.That(json.Contains("user"), "enum camel casing failed");
            Assert.That(json.Contains("max_tokens"), "custom prop name failed");
            Assert.That(json.Contains("top_p"), "custom prop name failed");
            Assert.That(json.Contains("frequency_penalty"), "custom prop name failed");
            Assert.That(json.Contains("presence_penalty"), "custom prop name failed");
        }

        [Test]
        public async Task ChatResponseDeserialization()
        {
            OpenAiClient client = getTestClient(@"ApiJson\ChatResponse.json");
            ChatCompletionResponse response = await client.CreateChat(this.userChatPrompt);
            Assert.NotNull(response);
            Assert.NotNull(response.Model);
            Assert.NotNull(response.Choices);
            Assert.NotNull(response.Choices[0].Message);
            Assert.AreEqual(response.Choices[0].FinishReason, FinishReason.Stop);
        }

        private OpenAiClient getTestClient(string jsonFile)
        {
            var mockHttpMessageHandler = new Mock<HttpMessageHandler>();
            StringContent content = new(File.ReadAllText(jsonFile));

            mockHttpMessageHandler.Protected()
                .Setup<Task<HttpResponseMessage>>("SendAsync",
                    ItExpr.IsAny<HttpRequestMessage>(), ItExpr.IsAny<CancellationToken>())
                .Callback((HttpRequestMessage msg, CancellationToken ct) => this.lastRequestMessage = msg)
                .ReturnsAsync(new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.OK,
                    Content = content
                });

            OpenAiClient client = new OpenAiClient("123", mockHttpMessageHandler.Object);
            return client;
        }
    }
}