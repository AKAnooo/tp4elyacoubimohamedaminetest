package com.example.test2;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.chat.ChatModel;

import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.nio.file.*;
import java.util.*;
import java.util.logging.*;

public class Test2 {

    private static void setupDebugLogs() {
        Logger log = Logger.getLogger("dev.langchain4j");
        log.setLevel(Level.FINE);
        ConsoleHandler h = new ConsoleHandler();
        h.setLevel(Level.FINE);
        log.addHandler(h);
    }

    public static void main(String[] args) {

        setupDebugLogs();
        System.out.println("----------- RAG Demo -----------");

        Path file = Paths.get("src/main/resources/rag.pdf");
        Document doc = FileSystemDocumentLoader.loadDocument(file, new ApacheTikaDocumentParser());

        var splitter = DocumentSplitters.recursive(140, 25);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel encoder = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> vectors = encoder.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(vectors, segments);

        String key = System.getenv("GEMINI_KEY");
        if (key == null || key.isBlank()) {
            throw new RuntimeException("Clé GEMINI_KEY manquante.");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(encoder)
                .embeddingStore(store)
                .maxResults(3)
                .minScore(0.30)
                .build();

        var memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .contentRetriever(retriever)
                .chatMemory(memory)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Pose ta question (exit pour quitter)");

        while (true) {
            System.out.print("> ");
            String q = scanner.nextLine().trim();
            if (q.equalsIgnoreCase("exit")) break;
            if (q.isEmpty()) continue;

            String answer = assistant.chat(q);
            System.out.println("\n" + answer + "\n");
        }
    }
}
