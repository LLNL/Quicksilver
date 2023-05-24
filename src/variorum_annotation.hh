#ifndef VARIORUM_ANNOTATION_H_INCLUDE
#define VARIORUM_ANNOTATION_H_INCLUDE
#include <stdio.h>
///@brief store the caller function's file, line and name.
#define CALL_INFO __FILE__, __LINE__, __func__
///@brief macro wrapper for actual functions, needed to store the call info.
#define VARIORUM_ANNOTATE_GET_NODE_POWER_JSON                                  \
  variorum_annotate_get_node_power_json(CALL_INFO)
///@brief macro wrapper for actual functions, needed to store the call info.
#define VARIORUM_ANNOTATE_GET_NODE_POWER_DOMAIN_INFO_JSON                      \
  variorum_annotate_get_node_power_domain_info_json(CALL_INFO)
///@brief wrapper for variorum node power json API.
///The Function writes the result in a json format 
///to a filename after the hostname.
void variorum_annotate_get_node_power_json(const char *file, int line,
                                           const char *function_name);
///@brief wrapper for variorum node power domain info json API.
///The Function writes the result in a json format 
///to a filename after the hostname.
void variorum_annotate_get_node_power_domain_info_json(
    const char *file, int line, const char *function_name);
#endif
