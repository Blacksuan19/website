---
layout: post
title: setting up nvim as an IDE
description: giving your text editor super powers
image: /assets/images/nvim.png
project: false
permalink: /blog/:title/
---

In my last post i went over my history of editors and the good, the bad and the
ugly about finally switching to vim, now its time to talk more about dem
configs boi.
My configs are a heavily modified version from [Optixal's Neovim Init.vim](https://github.com/Optixal/neovim-init.vim){:target="_blank"}, i have replaced some of the plugs and added my own bindings and autocompletion plugins.

#### Plugin Management
for plugins i left vim-plug as it is because well its great, gets the job done
and doesn't really interfere in any way, yes i do use a lot of plugins, however
most of them i consider necessary for a great IDE experience, out of the box
most of these functions are missing or hidden, i have added a comment for each
plugin explaining what it does.
<pre>
<code class="language-vim">
" ================= looks and GUI stuff ================== "

Plug 'vim-airline/vim-airline'                          " airline status bar
Plug 'vim-airline/vim-airline-themes'                   " airline themes
Plug 'ryanoasis/vim-devicons'                           " powerline like icons for NERDTree
Plug 'junegunn/rainbow_parentheses.vim'                 " rainbow paranthesis
Plug 'hzchirs/vim-material'                             " material color themes
Plug 'junegunn/goyo.vim'                                " zen mode
Plug 'amix/vim-zenroom2'                                " more focus in zen mode

" ================= Functionalities ================= "

" autocompletion using ncm2 (much lighter and faster than coc)
Plug 'ncm2/ncm2'
Plug 'roxma/nvim-yarp'
Plug 'ncm2/ncm2-bufword'
Plug 'ncm2/ncm2-path'
Plug 'filipekiss/ncm2-look.vim'
Plug 'fgrsnau/ncm-otherbuf'
Plug 'fgrsnau/ncm2-aspell'
Plug 'ncm2/ncm2-tern',  {'do': 'npm install'}
Plug 'ncm2/ncm2-pyclang'
Plug 'davidhalter/jedi-vim'
Plug 'ncm2/ncm2-jedi'
Plug 'ncm2/ncm2-vim' | Plug 'Shougo/neco-vim'
Plug 'ncm2/ncm2-ultisnips'
Plug 'ncm2/ncm2-html-subscope'
Plug 'ncm2/ncm2-markdown-subscope'

" markdown
Plug 'jkramer/vim-checkbox', { 'for': 'markdown' }
Plug 'dkarter/bullets.vim'                              " markdown bullet lists

" search
Plug 'wsdjeg/FlyGrep.vim'                               " project wide search
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all' }
Plug 'junegunn/fzf.vim'                                " fuzzy search integration

" snippets
Plug 'honza/vim-snippets'                               " actual snippets
Plug 'SirVer/ultisnips'                                 " snippets and shit

" visual
Plug 'majutsushi/tagbar'                                " side bar of tags
Plug 'scrooloose/nerdtree'                              " open folder tree
Plug 'jiangmiao/auto-pairs'                             " auto insert other paranthesis pairb
Plug 'alvan/vim-closetag'                               " auto close html tags
Plug 'Yggdroot/indentLine'                              " show indentation lines
Plug 'chrisbra/Colorizer'                               " show actual colors of color codes
Plug 'google/vim-searchindex'                           " add number of found matching search items

" languages
Plug 'sheerun/vim-polyglot'                             " many languages support
Plug 'tpope/vim-liquid'                                 " liquid language support

" other
Plug 'Chiel92/vim-autoformat'                           " an actually good and light auto formatter
Plug 'tpope/vim-commentary'                             " better commenting
Plug 'rhysd/vim-grammarous'                             " grammer checker
Plug 'tpope/vim-sensible'                               " sensible defaults
Plug 'lambdalisue/suda.vim'                             " save as sudo
Plug '907th/vim-auto-save'                              " auto save changes
Plug 'mhinz/vim-startify'                               " cool start up screen

</code>
</pre>
<br>
### superior auto completion
the original Optixal configs used deoplete for compeltion which is kinda heavy
and slow (this laptop is running an 10 APU) for this machine, i tried coc which
was good and super responsive but also still heavy espeically with python files
(jedi) i had many other options to pick from but i choose ncm2 becuase its fully
written in vim script and was meant for neovim so its a native enviroment for
it, and it works pretty good so far the popup comes up pretty fast and the
snippets work as intended. the good thing is i didnt really need to do a very
extensive setup because well this works by default i had to modify is the
mapping and other simple stuff which i could've used another plugin for.

{% include asciinema.html id="272608" %}
#### Visual changes:
just as usual it has to be material ocean themed, and vim being a popular editor
its already there and no need to reinvent the wheel, [kaicataldo's
themes](https://github.com/kaicataldo/material.vim){:target="_blank"} include it
and you can easily set it up, other visual changes include a modified air line
and a minimal NERDTree.

<pre>
<code class="language-vim">
let g:material_style='oceanic'
set background=dark
colorscheme vim-material
let g:airline_theme='material'
highlight Pmenu guibg=white guifg=black gui=bold
highlight Comment gui=bold
highlight Normal gui=none
highlight NonText guibg=none
autocmd ColorScheme * highlight VertSplit cterm=NONE ctermfg=Green ctermbg=NONE
</code>
</pre>
<br>
##### nifty tricks
these are vim built in options that arre not set uo by default or are set to not
so good defaults, i found most of these while trying to do a specfic task, i
used to use mswin behavior (makes nvim behave more like a regualr text editor)
but now i don't think i need it anymore, still haven't mastered the moved as i did
with good ol' plasma.

<pre>
<code class="language-vim">
" ==================== general config ======================== "

set termguicolors                                       " Opaque Background
set mouse=a                                             " enable mouse scrolling
set clipboard+=unnamedplus                              " use system clipboard by default

" ===================== Other Configurations ===================== "

filetype plugin indent on                               " enable indentations
set tabstop=4 softtabstop=4 shiftwidth=4 expandtab smarttab autoindent              " tab key actions
set incsearch ignorecase smartcase hlsearch             " highlight text while seaching
set list listchars=trail:»,tab:»-                       " use tab to navigate in list mode
set fillchars+=vert:\▏                                  " requires a patched nerd font (try furaCode)
set wrap breakindent                                    " wrap long lines to the width sset by tw
set encoding=utf-8                                      " text encoding
set number                                              " enable numbers on the left
set number relativenumber                               " relative numbering to current line (current like is 0 )
set title                                               " tab title as file file
set conceallevel=2                                      " set this so we womt break indentation plugin
set splitright                                          " open vertical split to the right
set splitbelow                                          " open horizontal split to the bottom
set tw=80                                               " auto wrap lines that are longer than that
set emoji                                               " enable emojis
let g:indentLine_setConceal = 0                         " actually fix the annoying markdown links conversion
au BufEnter * set fo-=c fo-=r fo-=o                     " stop annying auto commenting on new lines
set undofile                                            " enable persistent undo
set undodir=~/.nvim/tmp                                 " undo temp file directory
set ttyfast                                             " faster scrolling
set lazyredraw                                          " faster scrolling

</code>
</pre>
<br>
##### NERDTree
these configs are the best, i kanged the icons from some repo
which i can't remeber, it includes minimal ui (hides the top help text and other
stuff), and ignores .git and jeykll build files by default. it also quites when
you open a file so you can immediatly focus on the file
<pre>
<code class="language-vim">
let NERDTreeShowHidden=1
let NERDTreeShowLineNumbers=0
let g:NERDTreeDirArrowExpandable = ''
let g:NERDTreeDirArrowCollapsible = ''
let NERDTreeQuitOnOpen = 1
let NERDTreeMinimalUI = 1
let NERDTreeDirArrows = 1
let g:NERDTreeIgnore = [
\ '\.vim$',
\ '\~$',
\ '.git',
\ '_site',
\]
</code>
</pre>

#### Airline
airline is a fantastic status bar that shows important information like the
language, encoding and current mode youre in. my airline configs are pretty
simple as well, the only big difference compared to the defaults is the added
current line and percentage of the file at the right of the bar, also spellcheck
<pre>
<code class="language-vim">
" Airline
let g:airline_powerline_fonts = 0
let g:airline#themes#clean#palette = 1
call airline#parts#define_raw('linenr', '%l')
call airline#parts#define_accent('linenr', 'bold')
let g:airline_section_z = airline#section#create(['%3p%%  ',
\ g:airline_symbols.linenr .' ', 'linenr', ':%c '])
let g:airline_section_warning = ''
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#fnamemod = ':t'        " show only file name on tabs
</code>
</pre>
<br>
#### Other great features

- auto line indentation
- display tags via tag bar
- FZF integration
- grammer checking via Grammerous
- Snippets via Ultisnips
- startify when no buffer is open
- open images via feh


#### Key Mappings
The default keymaps are ok but they're missing some essential things like block
selection without going back to visual mode, fast split switching and split
rotation, so i made my own bindings for all these, some are from nvim help and
others are scattered from the web.<br>
i also fixed the annying block selection will copy the selected text to the
clipboard by using the void register for delete operations, this way deleting a
visual block will just delete it without adding it to the clipboard.
```vim
" use a different buffer for dd
nnoremap d "_d
vnoremap d "_d

" emulate windows copy, cut behavior
noremap <LeftRelease> "+y<LeftRelease>
noremap <C-c> "+y<CR>
noremap <C-x> "+d<CR>

" switch between splits using ctrl + {h,j,k,l}
tnoremap <C-h> <C-\><C-N><C-w>h
tnoremap <C-j> <C-\><C-N><C-w>j
tnoremap <C-k> <C-\><C-N><C-w>k
tnoremap <C-l> <C-\><C-N><C-w>l
inoremap <C-h> <C-\><C-N><C-w>h
inoremap <C-j> <C-\><C-N><C-w>j
inoremap <C-k> <C-\><C-N><C-w>k
inoremap <C-l> <C-\><C-N><C-w>l
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" select text via ctrl+shift+arrows in insert mode
inoremap <C-S-left> <esc>vb
inoremap <C-S-right> <esc>ve
```

#### Other shortcuts
- F3 for nerdtree
- f4 for tagbar
- f5 to rotate splits
- maybe more

and thats all from me, these settings do almost everything i need from an IDE
and keep me focused on the actual writing/coding part, and finally you can
checkout the file configuration file in my [dotfiles
repo](https://github.com/Blacksuan19/Dotfiles/blob/master/nvim/.config/nvim/init.vim).
