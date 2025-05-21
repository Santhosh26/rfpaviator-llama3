

def select_datastore(selected_option):
    #persistant directory for chromadb
    if selected_option:
        #select the chroma store    
        if "Fortify" in selected_option:
            persist_directory = "chromastore/Fortify_RFP"
        elif "NetIQ" in selected_option:
            persist_directory = "chromastore/NetIQ_RFP"
        elif "Extended ECM" in selected_option:
            persist_directory = "chromastore/ECM_RFP"
        elif "ALM.NET" in selected_option:
            persist_directory = "chromastore/ALM.NET_RFP"
        elif "Voltage" in selected_option:
            persist_directory = "chromastore/Voltage_RFP"
        elif "Arcsight" in selected_option:
            persist_directory = "chromastore/Arcsight_RFP"
        elif "Functional Testing" in selected_option:
            persist_directory = "chromastore/FUNCTIONAL_TESTING_RFP"
        elif "SMAX - AMX" in selected_option:
            persist_directory = "chromastore/SMAX-AMX_RFP"
        elif "Vertica" in selected_option:
            persist_directory = "chromastore/VERTICA_RFP"
        elif "IDOL" in selected_option:
            persist_directory = "chromastore/IDOL_RFP"
        elif "Performance Testing" in selected_option:
            persist_directory = "chromastore/PERFORMANCE_TESTING_RFP"
        elif "ALM Octane" in selected_option:
            persist_directory = "chromastore/OCTANE_RFP"
        elif "Exstream" in selected_option:
            persist_directory = "chromastore/Exstream_RFP"
        elif "Operations Bridge" in selected_option:
            persist_directory = "chromastore/OPERATIONS_BRIDGE_RFP"
        elif "NOM" in selected_option:
            persist_directory = "chromastore/NOM_RFP"
        elif "HCMX" in selected_option:
            persist_directory = "chromastore/HCMX_RFP"
        elif "CMS" in selected_option:
            persist_directory = "chromastore/CMS_RFP"
        elif "Media Management" in selected_option:
            persist_directory = "chromastore/Media_Management_RFP"
        elif "InfoArchive" in selected_option:
            persist_directory = "chromastore/InfoArchive_RFP"
        elif "Documentum" in selected_option:
            persist_directory = "chromastore/Documentum_RFP"
        elif "AppWorks" in selected_option:
            persist_directory = "chromastore/AppWorks_RFP"
        # elif "Documentum" in selected_option:
        #     persist_directory = "chromastore/Documentum_RFP"
        # elif "Documentum" in selected_option:
        #     persist_directory = "chromastore/Documentum_RFP"
        # elif "Documentum" in selected_option:
        #     persist_directory = "chromastore/Documentum_RFP"
        # elif "Documentum" in selected_option:
        #     persist_directory = "chromastore/Documentum_RFP"
        else:
            return None
        
        return persist_directory
    else:
        return None
