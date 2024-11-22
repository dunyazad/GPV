# Locate HDIAPI library
find_path(HDIAPI_INCLUDE_DIRS
          NAMES hidapi.h
          PATHS "C:/Program Files (x86)/hidapi/include/hidapi"
          NO_DEFAULT_PATH)

find_library(HDIAPI_LIBRARIES
             NAMES hidapi
             PATHS "C:/Program Files (x86)/hidapi/lib/")

if(HDIAPI_INCLUDE_DIRS AND HDIAPI_LIBRARIES)
    set(HDIAPI_FOUND TRUE)
else()
    set(HDIAPI_FOUND FALSE)
endif()

mark_as_advanced(HDIAPI_INCLUDE_DIRS HDIAPI_LIBRARIES)
