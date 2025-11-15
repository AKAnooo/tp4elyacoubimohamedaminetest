package com.example.test2;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface Assistant {

    @SystemMessage("""
        Tu interviens comme guide explicatif : tes réponses doivent être claires,
        structurées et compréhensibles. 
        Lorsque cela aide, fournis de petits exemples simples.
        Toutes les réponses doivent impérativement être rédigées en français.
        """)
    @UserMessage("Question de l’utilisateur : {{message}}")
    String chat(@V("message") String message);
}
