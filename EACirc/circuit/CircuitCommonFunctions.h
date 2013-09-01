#ifndef CIRCUITCOMMONFUNCTIONS_H
#define CIRCUITCOMMONFUNCTIONS_H

#include "EACglobals.h"
#include "GA1DArrayGenome.h"

/** get function type from node value
 * @param nodeValue
 * @return function constant
 */
unsigned char nodeGetFunction(GENOME_ITEM_TYPE nodeValue);

/** get argument from node value
 * @param nodeValue
 * @apram argumentNumber        (1-4, 1 being the most used, 4 being function constant)
 * @return argument 1 value
 */
unsigned char nodeGetArgument(GENOME_ITEM_TYPE nodeValue, int argumentNumber);

/** assign function constant to node value
 * @param nodeValue
 * @param function constant
 */
void nodeSetFunction(GENOME_ITEM_TYPE& nodeValue, unsigned char function);

/** assign argument to node value
 * @param nodeValue
 * @apram argumentNumber        (1-4, 1 being the most used, 4 being function constant)
 * @param argumentValue
 */
void nodeSetArgument(GENOME_ITEM_TYPE& nodeValue, int argumentNumber, unsigned char argumentValue);

/** read and discart the first connection from given conenctor mask
 * @param connectorMask     connector mask (relative or absolute) - is changed during run!
 * @param connection        position of the first connection from the left (0-31), undefined if no connection exists
 * @return is connection filled with connector?
 */
bool connectorsDiscartFirst(GENOME_ITEM_TYPE& connectorMask, int& connection);

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

/** returns neutral value for given function
 * @param function      function constant
 * @return neutral value
 */
unsigned char getNeutralValue(unsigned char function);

/** text describing function
 * @param function      function constant
 * @return describtion
 */
string functionToString(unsigned char function);

/** return label used in graph and code outputs, processes function layers
 * @param genome    source genome
 * @param layer     layer (-1 for pseudo-input, -2 for pseudo-output)
 * @param slot      slot number (0..size-1)
 * @return node description
 */
string getNodeLabel(GA1DArrayGenome<GENOME_ITEM_TYPE>& genome, int layer, int slot);

#endif // CIRCUITCOMMONFUNCTIONS_H
