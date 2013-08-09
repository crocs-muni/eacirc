#ifndef CIRCUITCOMMONFUNCTIONS_H
#define CIRCUITCOMMONFUNCTIONS_H

#include "EACglobals.h"
#include "GA1DArrayGenome.h"

/** get function type from node value
 * @param nodeValue
 * @return function constant
 */
unsigned char nodeGetFunction(GENOME_ITEM_TYPE nodeValue);

/** get argument 1 from node value
 * @param nodeValue
 * @return argument 1 value
 */
unsigned char nodeGetArgument1(GENOME_ITEM_TYPE nodeValue);

/** assign function constant to node value
 * @param nodeValue
 * @param function constant
 */
void nodeSetFunction(GENOME_ITEM_TYPE* nodeValue, unsigned char function);

/** assign argument 1 to node value
 * @param nodeValue
 * @param argument 1
 */
void nodeSetArgument1(GENOME_ITEM_TYPE* nodeValue, unsigned char argument1);

#endif // CIRCUITCOMMONFUNCTIONS_H
