#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

int main() {
    // Random training sets for XOR -- two inputs and one output
    printf("topology: 2 4 1\n");
    for (int i = 2000; i >= 0; --i) {
        int n1 = (int)(2.0 * rand() / double(RAND_MAX));
        int n2 = (int)(2.0 * rand() / double(RAND_MAX));
        int t = n1 ^ n2; // should be 0 or 1
        printf("in: %.1f %.1f\n", (double)n1, (double)n2);
        printf("out: %.1f\n", (double)t);
    }
}
