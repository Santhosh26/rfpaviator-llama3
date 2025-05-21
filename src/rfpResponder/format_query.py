def format_questions(question, product, solutions):

    temp = ""

    for solution in solutions:
        temp = temp + solution + ","
    

    
    append_question ="Does " + product +  " " + temp + "has" + " " + question #
    
    
    return append_question

# print(format_questions(question="what is the meaning of this?", product = "fortify", solutions=["SAST", "DAST", "SCA"]))