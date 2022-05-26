#include <iostream>

using namespace std;

// 一文轻松理解内存对齐
// https://cloud.tencent.com/developer/article/1727794

/*struct
{
    int x;
    char y;
} Test;

int main(int argc, char* args[])
{
    cout << sizeof(int) << endl;
    cout << sizeof(long) << endl;
    cout << sizeof(char) << endl;
    cout << sizeof(Test) << endl;

    cout << sizeof(void*) << endl;

    int* a = new int[5]{2, 1, 2, 3, 4};
    cout << a[-1] << endl;
    delete[] a;
    return 0;
}*/

#include <bits/stdc++.h>

#define ll long long
void debug()
{
#ifdef Acui
    freopen("data.in", "r", stdin);
    freopen("data.out", "w", stdout);
#endif
}

using namespace std;

// the alignment of all the allocated buffers
#define MALLOC_ALIGN 16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

int main()
{
    int* a = new int(5);
    std::cout << "a: " << a << std::endl;
    std::cout << "(size_t)a: " << (size_t)a << std::endl;
    std::cout << "(ll)a: " << (ll)a << std::endl;

    size_t totalsize = alignSize(12, 4);
    size_t totalsize2 = alignSize(17, 4);
    std::cout << "totalsize: " << totalsize << std::endl;
    std::cout << "totalsize: " << totalsize2 << std::endl;
    void* data = fastMalloc(totalsize);
    fastFree(data);
    std::cout << "data:         " << data << std::endl;
    std::cout << "(void*)data:  " << (void*)data << std::endl;
    std::cout << "(int*)data:   " << (int*)data << std::endl;
    std::cout << "(float*)data: " << (float*)data << std::endl;
    return 0;
}