VERBOSE=1
if [[ $# -ge 4 ]]; then
    inpdir=$1;
    filename=$2;
    outdir=$4;
    lineno=$3;
    mkdir -p 'tmp';
    cp $inpdir'/'$filename 'tmp/'$filename;
    # 删除旧的同名输出目录
    if [[ -d "$filename" ]]; then
      rm -rf "$filename";
    fi
    ./joern/joern/joern-parse tmp $filename;
    rm 'tmp/'$filename
    # 移动结果到目标目录
    if [[ -d "$filename/tmp/$filename" ]]; then
      mkdir -p "$outdir";
      cp "$filename/tmp/$filename/nodes.csv" "$outdir/" 2>/dev/null;
      cp "$filename/tmp/$filename/edges.csv" "$outdir/" 2>/dev/null;
      rm -rf "$filename";
    fi
else
  echo 'Wrong Argument!.'
fi
