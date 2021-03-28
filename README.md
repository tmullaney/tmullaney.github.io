# tmullaney.github.io

## Setup
This site uses the [Jekyll](https://jekyllrb.com/docs/) static site generator. 
```
# Install Ruby
brew install ruby
export GEM_HOME="$HOME/.gem"

# Add this line to ~/.zshrc too
# export GEM_HOME="$HOME/.gem"

# Install jekyll
gem install jekyll bundler

# Special workaround
# See: https://github.com/ffi/ffi/issues/653#issuecomment-698563530
# original: ffi (1.9.17)
gem install ffi -v '1.13.1' -- --with-cflags="-Wno-error=implicit-function-declaration"

# Install gem requirements
bundle install
```

## Run locally
```
bundle exec jekyll serve
```