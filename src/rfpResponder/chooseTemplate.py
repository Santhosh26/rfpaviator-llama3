from langchain import hub

#Original

    # common_template = """You are an expert {role} with extensive experience in Opentext {product_name}.
    # You are appointed for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    # If you don't know the answer, just say that you don't know. 
    # When answering, begin your responses with either 'Yes,' or 'No,' depending on sentiment of the response. 
    # offer an explanation to your answer. 
    # Refrain from engaging in extended conversations or discussions beyond the scope of the direct question. 
    # Don't direct users to documentation or other sources.
    # In your answer or response, dont mention about the following documentation and datasheet references.

    # Question: {{question}}
    
    # Context: {{context}}

    # Answer:
    # """


def returnTemplate(product, btDetailed):
    # Common prompt template
    common_template = """You are an expert {role} with extensive experience in Opentext {product_name}. Answer the following question based on the provided context. If you don't have enough information to answer, state that you don't know.

Instructions to be followed:

Begin your response with either "Yes," or "No," (without any additional characters or formatting), depending on the sentiment of your answer. Use "Yes," for positive sentiment and "No," for negative sentiment.
Always provide an explanation for your answer.
Do not mention that your answer is based on retrieved information.
Stick to answering the specific question asked without engaging in extended discussions.
Do not direct users to external documentation or sources.
Do not include any formatting characters (like asterisks, underscores, or markdown syntax) in your response.
Context: {{context}}

Question: {{question}}

Answer:
    """
    
    # Product-specific configurations
    product_configs = {

        "Fortify": "Application Security architect, Fortify",
        "NetIQ": "Identity and access management architect, NetIQ",
        "Arcsight": "Security Operations architect, Arcsight",
        "Voltage": "Data Security architect, Voltage",
        "Extended ECM": "Content Management architect, Extended ECM",
        "AppWorks": "Process Automation architect, AppWorks",
        "Documentum": "Content Management architect, Documentum",
        "Exstream": "Customer Communications Management expert, Exstream",
        "InfoArchive": "Enterprise Archiving, InfoArchive",
        "Media Management": "Digital Asset Management architect, Media Management",
        "CMS": " web content management, CMS",
        "HCMX": "Hybrid Cloud management architect, HCMX",
        "NOM": "Network Operations Management architect, NOM",
        "Operations Bridge": "Customer experience architect, Operations Bridge",
        "SMAX - AMX": "service management architect, Extended SMAX - AMX",
        "ALM.NET": "Application Lifecycle Management architect, ALM.NET",
        "ALM Octane": "Application Lifecycle Management architect, ALM Octane",
        "Performance Testing": "Performance Testing architect, Performance Testing",
        "Functional Testing": "Functional Testing architect, Functional Testing",
        "IDOL": "Intelligent Data Operating Layer architect, IDOL" ,
        "Vertica": "analytical database expert, Vertica",
    }

    # Select the role and product name based on the product argument
    role, product_name = product_configs.get(product, ("", "")).split(", ")
    
    # Fill in the common template with the product-specific details
    template = common_template.format(role=role, product_name=product_name)
    
    # Add detailed instruction if btDetailed is True
    if btDetailed:
        template += " Be very detailed in your answer, providing comprehensive information and insights."
        
    return template


# returnTemplate(product="Arcsight", btDetailed=None)