#include "../lznn_tools.cpp"

int main(void)
{
    Matrix m(10, Vector(10, 0.0));
    Tools::dump(m, "dumpMatrix.txt", "test");
    Tools::dump(m[0], "dumpVector.txt", "test");
    return 0;
}