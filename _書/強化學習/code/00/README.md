```
pip install gymnasium[box2d] => 失敗
```

Box2D\Common\b2Math.h(85) : Warning 509: as it is shadowed by b2Vec2::operator ()(int32) const.
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

Windows 需先下載 build-tools

* https://visualstudio.microsoft.com/visual-cpp-build-tools/


download swig

* https://www.swig.org/download.html


解開，看

file:///C:/Users/user/Downloads/swig-4.1.1/Doc/Manual/Windows.html


安裝 swigwin:

https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.1.1/swigwin-4.1.1.zip/download?use_mirror=nchc


然後再次

```
pip install gymnasium[box2d]
```

還是失敗 ...

不過別用 box2d 顯示就會成功，改用 human !
