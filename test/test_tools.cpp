#include "../tools.cpp"

int main(void)
{
    Matrix m(10, Vector(10, 0.0));
    Tools::dump(m, "dumpMatrix.txt");
    Tools::dump(m[0], "dumpVector.txt");
    return 0;
}