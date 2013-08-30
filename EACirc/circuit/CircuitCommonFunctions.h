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

/** transform connector mask relative->absolute (when saving)
 * @param relativeMask
 * @param slot                  current slot within the layer
 * @param sizePreviousLayer     number of function nodes in previous layer (when to wrap around)
 * @param connectorWidth        number of applicable connectors (numConnectors, if corresponding to it)
 * @return absolute connector mask
 */
GENOME_ITEM_TYPE relativeToAbsoluteConnectorMask(GENOME_ITEM_TYPE relativeMask, int slot, int sizePreviousLayer, int connectorWidth);

/** transform connector mask absolute->relative (when loading)
 * @param absoluteMask
 * @param slot                  current slot within the layer
 * @param sizePreviousLayer     number of function nodes in previous layer (when to wrap around)
 * @param connectorWidth        number of applicable connectors (numConnectors, if corresponding to it)
 * @return relative connector mask
 */
GENOME_ITEM_TYPE absoluteToRelativeConnectorMask(GENOME_ITEM_TYPE absoluteMask, int slot, int sizePreviousLayer, int connectorWidth);

#endif // CIRCUITCOMMONFUNCTIONS_H
