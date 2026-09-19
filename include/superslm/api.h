#ifndef SUPERSLM_API_H
#define SUPERSLM_API_H
/* Export slot for the engine symbols a module in another shared library calls. Empty by
 * default. A build that shares one engine copy between shared libraries defines SUPERSLM_API
 * before any SuperSLM header is read, normally on the compiler command line: the platform
 * export attribute when compiling the engine's library, the import attribute when compiling
 * a consumer. In Unreal, PublicDefinitions.Add("SUPERSLM_API=<ENGINEMODULE>_API") on the
 * engine module does both. */
#ifndef SUPERSLM_API
#define SUPERSLM_API
#endif
#endif
