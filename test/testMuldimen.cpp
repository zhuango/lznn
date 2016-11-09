#include<vector>
#include<iostream>
using namespace std;
#include "../lznn_types.h"
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
int main(void)
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
    return 1;
    
}
