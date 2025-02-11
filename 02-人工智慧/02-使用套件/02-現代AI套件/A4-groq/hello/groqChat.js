import Groq from "npm:groq-sdk"

let question =  Deno.args.join(' ')

const groq = new Groq({
    apiKey: Deno.env.get("GROQ_API_KEY")
});

async function getGroqChatCompletion() {
    return groq.chat.completions.create({
        messages: [
            {
                role: "user",
                content: question // "Explain the importance of fast language models"
            }
        ],
        model: "llama3-8b-8192"
    });
}

async function main() {
    const chatCompletion = await getGroqChatCompletion();
    // Print the completion returned by the LLM.
    console.log(chatCompletion.choices[0]?.message?.content || "");
}

main()
