import pandas as pd
import time
# from src.compare_comp.rager import qa_rag
from src.compare_comp.choose_product import chooseCompareProdcucts
from src.compare_comp.preplexRag_search import perplex
# from rager import qa_rag
# from  choose_product import chooseCompareProdcucts
def asses_competition(questions,selected_option):
    start_time = time.time()

    products = chooseCompareProdcucts(selectedOption=selected_option)
    
    product_percentage_table = []

    for product in products:
        yes_count = 0
        total_count = 0  # Total number of questions processed for each product
        no_count = 0
        for question in questions:
            # print("question: " + question)
            response = perplex(question, product)
            # print("Answer: " + response['result'])
            if response.startswith('Yes'):
                yes_count += 1
            else:
                no_count += 1

            total_count += 1  # Increment for each question processed

        yes_percentage = f"{(yes_count / total_count) * 100:.2f}%" if total_count else 0
        product_percentage_table.append({"Product": product, "Complaince": yes_percentage})

    end_time = time.time()
    time_taken = (end_time - start_time) / 60  # Convert to minutes

    # Convert to DataFrame
    results_df = pd.DataFrame(product_percentage_table)
    return time_taken, results_df

# print (asses_competition(questions=['The solution should support over 70 programming languages (include scripting languages) and provide IDE plugins for scanning and remediation.'], selected_option = "Fortify" ))