cp -r $1 $1_bak 

# perl -i.bak.markdown  -p \
perl -i -p \
  -e 's/(\s*)\\\[/\n$1\$\$/g;' \
  -e 's/\\\]/\$\$\n/g;' \
  -e 's/\\\(\s*/ \$/g;' \
  -e 's/\s*\\\)/\$ /g;' \
  $1/*.md