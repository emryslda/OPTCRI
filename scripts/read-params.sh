# Reads OPTCRI parameter file and assigns the variables
# ADOPTED FROM CHIMERE model scripts 

echo '# DO NOT CHANGE!' > ${optcriparash}

echo >> ${optcriparash}

while read l; do echo "$l" >> ${optcriparash}; done < <(${my_awk} -v rind=1 '{
   # seek for the variable
   for (i=1; i<=NF-2; i++) {
      if (substr($i,1,1) == "#") break;
      if (substr($i,1,2) == "[$" && substr($i,length($i),1) == "]") {
         # found the variable
         var = substr($i,3,length($i)-3);
         # now need to get the values right of the ':' sign
         # seek for ':'
         for (j=i+1; j<=NF; j++) {
            if ($j == ":") {
               # found ":"; get list of comma-separated values
               s="";
               for (v=j+1; v<=NF; v++) {
                  if (substr($v,1,1) == "#") break;
                  s = s$v" ";
               }
               split(s, slist, ",");
               # Return parameters for the index requested
               ind = rind ;
               if (length(slist) < rind) ind = length(slist);
               val = slist[ind];
               # left-trim
               gsub(/^[ \t]+/, "", val);
               # right-trim
               gsub(/[ \t]+$/, "", val);

               # date arithmetic if required
               # we need smth like `date -u -d "$di 2 days ago" +%Y%m%d`
               xx=0;
               while (xx == 0) {
                  # 1. get leftmost string inside (( ))
                  nm = match(val, /(\(\()[^((]+(\)\))/, arr);
                  if (nm == 0) break;
                  # 2. Remove the parentheses
                  sexpr = substr( arr[0], 3, length(arr[0])-4 );
                  # 3. split by +/- and construct the bash date expression
                  newexp = "";
                  split(sexpr, explist, "+");
                  if (length(explist) == 2) newexp = ("`date -u -d \"" explist[1] " " explist[2] " days\" +%Y%m%d`");
                  split(sexpr, explist, "-");
                  if (length(explist) == 2) newexp = ("`date -u -d \"" explist[1] " " explist[2] " days ago\" +%Y%m%d`");
                  # 4. replace the orignal expression by the new one
                  if (length(newexp) > 0) sub(/(\(\()[^((]+(\)\))/, newexp, val);
               }
               print ("export "var"=\x27"val"\x27");
               break;
            }
         }
      }
   }
   }' $optcriparams)
