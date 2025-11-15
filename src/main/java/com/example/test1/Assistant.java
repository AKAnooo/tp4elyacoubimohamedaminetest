package com.example.test1;

import dev.langchain4j.service.SystemMessage;

public interface Assistant {

    @SystemMessage("""
        Tu fonctionnes comme un assistant utilisant un système RAG.
        Tes réponses doivent uniquement s’appuyer sur le contenu du PDF fourni,
        sans ajouter d'informations extérieures.
    """)
    String chat(String userMessage);
}
