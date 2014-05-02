#ifndef MAIN_H
#define MAIN_H

/**
  * Main method
  * - parses CLI arguments
  * - creates single EACirc object
  * - prepares everything and starts the computation
  */
int main(int argc, char **argv);

/** Runtime environment tests
 * - check limits for UCHAR
 */
void testEnvironment();

#endif // MAIN_H
