// Deliberately broken kernel: several distinct, unambiguous sema errors in one file, used to
// prove `--sema` reports all of them (rather than stopping at the first) and returns promptly.
struct Point {
    int x;
    int y;
};

__global__ void broken(int *out) {
    // Reference to an identifier that was never declared (E301).
    int a = undeclared_variable;

    // Assigning a struct value where an int is expected (E300).
    struct Point p;
    int b = p;

    // Redefining a name already bound in the same scope (E302).
    int c = 1;
    int c = 2;

    out[0] = a + b + c;
}
