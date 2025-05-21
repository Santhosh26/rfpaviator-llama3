def showProductgroup():
    prodcutGroupList = ["AAI", "ADM", "CyberSecurity", "Content", "IT Operations Management" ]
    return prodcutGroupList

def showProducts(selectedProductGroup):
    if selectedProductGroup == "CyberSecurity":
        prodcutList = ["Fortify", "NetIQ", "Arcsight", "Voltage"]
        
    elif selectedProductGroup == "Content":
        prodcutList = ["AppWorks", "Documentum", "Exstream", "Extended ECM", "InfoArchive", "Media Management" ]
        
    elif selectedProductGroup == "IT Operations Management":
        prodcutList = ["CMS", "HCMX", "NOM", "Operations Bridge", "SMAX - AMX"]
    elif selectedProductGroup == "ADM":
        prodcutList = ["ALM.NET", "ALM Octane", "Performance Testing", "Functional Testing"]
    elif selectedProductGroup == "AAI":
        prodcutList = ["IDOL", "Vertica"]

    return prodcutList


def solutionSelection(selectedProduct):

    if selectedProduct == "Fortify":
        subProductList = ["SAST", "DAST", "Sonatype Nexus IQ"]
    elif selectedProduct == "NetIQ":
        subProductList = ["Access Management", "IGA", "PAM", "Policy Orchestration"]
    elif selectedProduct == "Arcsight":
        subProductList = ["ESM", "Logger", "Recon", "Intelligence", "Connectors", "SOAR"]
    elif selectedProduct == "Voltage":
        subProductList = ["Voltage DAM", "Structured Data Manager", "Secure Data", "Sentry"]
    else:
        return
    # if selectedProduct == "Extended ECM":
    #     #subProductList = ["x", "y", "z"]
    #     return
    # if selectedProduct == "ITOM":
    #     subProductList = ["Operations Bridge", "SMAX - AMX", "CMS", "NOM", "HCMX"]
         
    # if selectedProduct == "ADM":
    #     subProductList = ["ALM.NET", "ALM OCTANE", "Performance Testing", "Functional Testing"]
        
    if selectedProduct == "AAI":
        subProductList = ["IDOL", "VERTICA"]
        
    return subProductList
# def returnSubProduct(subProdcutlist)
    
#     for item in selected_subOption:
#     onSelectedSuboption = st.checkbox (label=item, key = item, value = False)
#     st.write(item)
#     st.write(onSelectedSuboption)