package com.example.test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class Test4SansRag {

    public static void main(String[] args) {

        System.out.println("------------ Test 4 : RAG déclenché uniquement si nécessaire ------------");

        Document source = FileSystemDocumentLoader.loadDocument(
                Paths.get("src/main/resources/QCM_MAD-AI_COMPLET.pdf"),
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> chunks = DocumentSplitters.recursive(280, 35).split(source);

        EmbeddingModel encoder = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = encoder.embedAll(chunks).content();

        EmbeddingStore<TextSegment> repository = new InMemoryEmbeddingStore<>();
        repository.addAll(embeddings, chunks);

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(repository)
                .embeddingModel(encoder)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Décide si le RAG doit être utilisé en fonction du sujet de la question
        QueryRouter router = query -> {
            String inspection = """
                    La question porte-t-elle sur la RAG, les embeddings, les LLM,
                    le fine-tuning ou tout autre concept lié à l’IA ?
                    Réponds uniquement "oui" ou "non".

                    %s
                    """.formatted(query.text());

            String decision = model.chat(inspection).toLowerCase();
            return decision.contains("oui") ? List.of(retriever) : List.of();
        };

        RetrievalAugmentor augmenter = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        var assistant = AiServices.builder(AssistantLimite.class)
                .chatModel(model)
                .retrievalAugmentor(augmenter)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Pose une question (tape 'exit' pour quitter) : ");
            String userInput = scanner.nextLine();
            if (userInput.equalsIgnoreCase("exit")) break;

            System.out.println(assistant.chat(userInput));
        }
    }
}
