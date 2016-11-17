#include<vector>
#include<iostream>
using namespace std;
#include "../src/lznn_types.h"
#include <cmath>

class T
{
    public:
        T(int a)
        :T(a, 10)
        {
        }
        T(int a, int b)
        :va(Vector(a, 10.0)), b(b)
        {
        }
        Vector va;
        int b;
    private:
};
void testMuldimen()
{
    vector< vector<int> > mutimen;
    vector<int> single1;
    vector<int> single2;
    for(int i = 0; i < 10; i++)
    {
        single1.push_back(i);
        single2.push_back(i);
    }
    mutimen.push_back(single1);
    mutimen.push_back(single2);

    cout << mutimen[1][2] << endl;

    Matrix a = Matrix(2, Vector(3, 0.0));
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }

    T t(3);
    cout << t.va.size() << " " << t.b << endl;
}

class A
{
    public:
        A(int a)
        :a(a)
        {
            cout << a << " in A" << endl;
        }
        virtual void print()
        {
            cout << "A" << endl;
        }
    protected:
        int a;
};
class B: public A
{
    public:
        B(int a)
        :A(a)
        {

        }
        void print()
        {
            cout << "B" << endl;
        }
        int getA()
        {
            return a;
        }
};
double distance(double a, double b)
{
    return 1.0;
}
int main(void)
{
    A *a = new B(1);
    a->print();
    typedef double (*DistanceFunc)(double, double);
    DistanceFunc func = distance;
    cout << sqrt( pow(3, 2.0) ) << endl;
    cout << func(1, 2) << endl;
    return 1;
}
