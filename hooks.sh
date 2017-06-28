# Setup git hooks
chmod -R u+x hooks/*
for f in hooks/*; do
    rm -rf .git/$f
done
ln hooks/* .git/hooks/
